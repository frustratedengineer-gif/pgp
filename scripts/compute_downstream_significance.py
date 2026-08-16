#!/usr/bin/env python
"""
Week-6 reviewer gap: every downstream-QA claim (week6_downstream_qa*.md,
week6_ranked_eviction_sweep.md) was reported as a raw mean EM/F1 with no
confidence interval or significance test -- unlike the Week-3/4 C-index
claims, which got a full bootstrap treatment (week4_significance.md).
This closes that gap using the exact same method (paired bootstrap, see
src/memorylife/evaluation/downstream_significance.py), applied to
predictions already collected -- no new LLM calls, no new cost.

Two families of comparisons, both on the matched 145-question sample
(120 LoCoMo + 25 LongMemEval) used throughout the Week-6 confirmation runs:

1. Fix #2 (ranked eviction), within results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json:
   ours_utility vs. lru/fifo/ours/no_forget -- is "beats lru" and
   "matches the no_forget ceiling" real, or within noise at n=120/25?

2. Fix #1 (TTL quantile), across the three quantile pilot raw files:
   ours@Q0.2 vs ours@Q0.5, ours@Q0.1 vs ours@Q0.5, ours@Q0.1 vs ours@Q0.2
   -- does moving the cutoff off the median significantly improve ours
   on real EM/F1, not just the free evidence-retention proxy?

    python scripts/compute_downstream_significance.py
"""
import argparse
import json
from pathlib import Path

from memorylife.evaluation.downstream_significance import bootstrap_paired_mean_diff


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def by_policy_question(rows: list[dict], policy: str) -> dict[tuple, dict]:
    return {(r["benchmark"], r["conversation_id"], r["question"]): r
            for r in rows if r["policy"] == policy}


def paired_metric_arrays(rows: list[dict], policy_a: str, policy_b: str, metric: str,
                          benchmark: str | None = None) -> tuple[list[float], list[float]]:
    a_by_key = by_policy_question(rows, policy_a)
    b_by_key = by_policy_question(rows, policy_b)
    keys = sorted(set(a_by_key) & set(b_by_key))
    if benchmark:
        keys = [k for k in keys if k[0] == benchmark]
    assert keys, f"no overlapping questions for {policy_a} vs {policy_b} (benchmark={benchmark})"
    assert len(keys) == len({k for k in a_by_key if not benchmark or k[0] == benchmark}), \
        f"{policy_a}/{policy_b} question sets don't fully match -- alignment bug, not just missing coverage"
    return [a_by_key[k][metric] for k in keys], [b_by_key[k][metric] for k in keys]


def run_comparison(rows, policy_a, policy_b, benchmark, n_boot, seed) -> dict:
    out = {"benchmark": benchmark, "a": policy_a, "b": policy_b}
    for metric in ("em", "f1"):
        va, vb = paired_metric_arrays(rows, policy_a, policy_b, metric, benchmark)
        res = bootstrap_paired_mean_diff(va, vb, n_boot=n_boot, seed=seed)
        out[metric] = res
    return out


def format_row(r: dict) -> str:
    em, f1 = r["em"], r["f1"]
    return (f"| {r['benchmark']} | {r['a']} vs {r['b']} | n={em['n']} | "
            f"{em['mean_diff']:+.4f} [{em['ci_low']:+.4f}, {em['ci_high']:+.4f}] | p={em['p_value_one_sided']:.3f} | "
            f"{f1['mean_diff']:+.4f} [{f1['ci_low']:+.4f}, {f1['ci_high']:+.4f}] | p={f1['p_value_one_sided']:.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="results/raw")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw_dir = Path(args.raw_dir)

    header = ["| Benchmark | Comparison (a vs b) | N | EM diff (a-b), 95% CI | p (a<=b) | F1 diff (a-b), 95% CI | p (a<=b) |",
              "|---|---|---|---|---|---|---|"]

    # --- Fix #2: ranked (utility) eviction vs. everything else, at Q=0.2 ---
    ranked_rows = load_rows(raw_dir / "week6_downstream_qa_raw_q0.2_ranked_pilot.json")
    fix2_lines = list(header)
    for benchmark in ("locomo", "longmemeval"):
        for other in ("lru", "fifo", "ours", "no_forget"):
            r = run_comparison(ranked_rows, "ours_utility", other, benchmark, args.n_boot, args.seed)
            fix2_lines.append(format_row(r))

    # --- Fix #1: does the TTL-quantile fix itself significantly help `ours`? ---
    q05 = load_rows(raw_dir / "week6_downstream_qa_raw_q0.5_pilot_control.json")
    q02 = load_rows(raw_dir / "week6_downstream_qa_raw_q0.2_pilot.json")
    q01 = load_rows(raw_dir / "week6_downstream_qa_raw_q0.1_pilot.json")
    # normalize into one list tagged by quantile so paired_metric_arrays can select policy="ours@Q"
    combined = []
    for tag, rows in (("ours@Q0.5", q05), ("ours@Q0.2", q02), ("ours@Q0.1", q01)):
        for r in rows:
            if r["policy"] == "ours":
                combined.append({**r, "policy": tag})

    fix1_lines = list(header)
    for benchmark in ("locomo", "longmemeval"):
        for a, b in (("ours@Q0.2", "ours@Q0.5"), ("ours@Q0.1", "ours@Q0.5"), ("ours@Q0.1", "ours@Q0.2")):
            r = run_comparison(combined, a, b, benchmark, args.n_boot, args.seed)
            fix1_lines.append(format_row(r))

    md = "\n".join([
        "# Week-6 downstream QA: bootstrap significance (paired, same-question resampling)",
        "",
        f"n_boot={args.n_boot}, seed={args.seed}. `p (a<=b)` is the fraction of bootstrap",
        "replicates where policy `a` did NOT beat policy `b` -- small means `a` reliably",
        "beats `b`; a CI straddling 0 alongside p near 0.5 means statistically",
        "indistinguishable (the relevant read for an \"ours_utility matches the ceiling\" claim).",
        "",
        "## Fix #2: does utility-ranked eviction (`ours_utility`) really beat lru / match the ceiling?",
        "",
        *fix2_lines,
        "",
        "## Fix #1: does moving the TTL cutoff off the median significantly help `ours`?",
        "",
        *fix1_lines,
    ])

    out_dir = Path(args.out_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "week6_downstream_significance.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
