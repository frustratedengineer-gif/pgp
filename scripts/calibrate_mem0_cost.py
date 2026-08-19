#!/usr/bin/env python
"""
One-off calibration run (not the real eval): indexes ONE LoCoMo
conversation into Mem0 turn-by-turn, checking real OpenRouter balance
periodically (via /api/v1/credits) so it aborts safely instead of
running the account to zero if the per-turn cost is higher than
estimated. Reports measured $/turn so scripts/eval_mem0_baseline.py's
full-scope cost can be extrapolated from a real number, not a guess.

    OPENROUTER_API_KEY=... python scripts/calibrate_mem0_cost.py --conversation conv-30 --safety-floor 0.15
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from mem0_wrapper import add_turn, build_memory  # noqa: E402


def get_balance(api_key: str) -> float:
    r = requests.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {api_key}"})
    d = r.json()["data"]
    return d["total_credits"] - d["total_usage"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo-path", default="data/raw/locomo10.json")
    ap.add_argument("--conversation", default="conv-30")
    ap.add_argument("--qdrant-path", default="/tmp/mem0_calibration_qdrant")
    ap.add_argument("--check-every", type=int, default=10, help="check balance every N turns")
    ap.add_argument("--safety-floor", type=float, default=0.30, help="abort if balance drops below this many dollars")
    ap.add_argument("--max-turns", type=int, default=60, help="calibration cap -- don't need the whole conversation to get a stable $/turn rate")
    args = ap.parse_args()

    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set.")

    shutil.rmtree(args.qdrant_path, ignore_errors=True)
    mem = build_memory("calibration", args.qdrant_path)

    data = json.loads(Path(args.locomo_path).read_text(encoding="utf-8"))
    conv = next(c for c in data if c["sample_id"] == args.conversation)["conversation"]
    session_keys = sorted((k for k in conv if k.startswith("session_") and not k.endswith("date_time")),
                          key=lambda k: int(k.split("_")[1]))

    balance_start = get_balance(api_key)
    print(f"Starting balance: ${balance_start:.4f}")

    n_turns = 0
    aborted = False
    t0 = time.time()
    for sess_key in session_keys:
        date_str = conv.get(f"{sess_key}_date_time", "")
        for turn in conv[sess_key]:
            add_turn(mem, "calibration-conv", turn["speaker"], turn["text"], f"[{date_str}]")
            n_turns += 1
            if n_turns % args.check_every == 0:
                balance = get_balance(api_key)
                spent = balance_start - balance
                rate = spent / n_turns
                print(f"  {n_turns} turns indexed, ${spent:.4f} spent so far (${rate:.5f}/turn), "
                      f"${balance:.4f} remaining")
                if balance < args.safety_floor:
                    print(f"ABORTING: balance ${balance:.4f} below safety floor ${args.safety_floor:.2f}")
                    aborted = True
                    break
            if n_turns >= args.max_turns:
                print(f"Reached calibration cap of {args.max_turns} turns -- stopping (this is a "
                      f"calibration run, not the full eval).")
                aborted = True
                break
        if aborted:
            break

    balance_end = get_balance(api_key)
    spent = balance_start - balance_end
    elapsed = time.time() - t0
    rate = spent / n_turns if n_turns else 0.0

    print(f"\n=== Calibration result ===")
    print(f"Conversation: {args.conversation} ({'aborted early' if aborted else 'completed'})")
    print(f"Turns indexed: {n_turns}")
    print(f"Spent: ${spent:.4f}")
    print(f"$/turn: ${rate:.5f}")
    print(f"Wall-clock: {elapsed:.1f}s ({elapsed/max(n_turns,1):.2f}s/turn)")
    print(f"Remaining balance: ${balance_end:.4f}")
    print(f"\nExtrapolated cost for all 5,882 LoCoMo turns (10 conversations): ${rate * 5882:.2f}")

    md = (
        "# Mem0 indexing cost calibration (GPT-4o via OpenRouter)\n\n"
        f"Real run: `{args.conversation}`, capped at the first {args.max_turns} turns (a calibration "
        "run, not the full eval), Mem0's `openai` LLM provider auto-routed through OpenRouter to "
        "GPT-4o.\n\n"
        "| Metric | Value |\n|---|---|\n"
        f"| Turns indexed | {n_turns} |\n"
        f"| Starting balance | ${balance_start:.4f} |\n"
        f"| Ending balance | ${balance_end:.4f} |\n"
        f"| Spent | ${spent:.4f} |\n"
        f"| $/turn | ${rate:.5f} |\n"
        f"| Wall-clock | {elapsed:.1f}s ({elapsed/max(n_turns,1):.2f}s/turn) |\n"
        f"| Extrapolated cost for all 5,882 LoCoMo turns (10 conversations) | ${rate * 5882:.2f} |\n"
    )
    out_path = Path("results/tables/week6_mem0_cost_calibration.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
