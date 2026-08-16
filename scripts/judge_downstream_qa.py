#!/usr/bin/env python
"""
Week-6 reviewer gap: `llm_judge_score` (src/memorylife/evaluation/qa_metrics.py)
was built specifically because EM penalizes correct-but-differently-worded
answers (e.g. "2023-05-07" vs "7 May 2023" -- an EM=0 pair that actually
appears verbatim in results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json)
but was never actually run against any Week-6 predictions. This judges
predictions ALREADY COLLECTED (no new retrieval/answering, no new
run_downstream_qa_eval.py call) -- one short LLM call per (question,
prediction, reference) triple, much cheaper than a full QA run since
there's no retrieved-memories block in the prompt.

    OPENROUTER_API_KEY=... python scripts/judge_downstream_qa.py \
        --input results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json
"""
import argparse
import hashlib
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memorylife.evaluation.qa_metrics import llm_judge_score


def _cache_key(model: str, question: str, prediction: str, reference: str) -> str:
    return hashlib.sha256(f"{model}||{question}||{prediction}||{reference}".encode()).hexdigest()


def judge_one(row: dict, model: str, cache_dir: Path | None) -> tuple[float, dict]:
    cache_path = None
    if cache_dir is not None:
        key = _cache_key(model, row["question"], row["prediction"], str(row["reference"]))
        cache_path = cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return cached["judge"], cached.get("usage", {})

    score, usage = llm_judge_score(row["question"], row["prediction"], str(row["reference"]), model=model)

    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"judge": score, "usage": usage}))
    return score, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw predictions JSON from run_downstream_qa_eval.py")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache-dir", default="artifacts/llm_cache/judge")
    ap.add_argument("--llm-model", default="openai/gpt-4o")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set in environment -- export it and re-run.")

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(judge_one, r, args.llm_model, cache_dir) for r in rows]
        for row, fut in zip(rows, futures):
            score, usage = fut.result()
            row["judge"] = score
            for k_, v in usage.items():
                if k_ in usage_totals:
                    usage_totals[k_] += v
            usage_totals["calls"] += 1

    disagreements = [r for r in rows if r["em"] == 0.0 and r["judge"] == 1.0]

    summary: dict[tuple, dict] = {}
    for r in rows:
        key = (r["benchmark"], r["policy"])
        summary.setdefault(key, {"em": [], "f1": [], "judge": []})
        summary[key]["em"].append(r["em"])
        summary[key]["f1"].append(r["f1"])
        summary[key]["judge"].append(r["judge"])

    md_lines = ["| Benchmark | Policy | Mean EM | Mean F1 | Mean Judge (LLM-graded correct) | N |",
                "|---|---|---|---|---|---|"]
    for (benchmark, policy), d in sorted(summary.items()):
        md_lines.append(f"| {benchmark} | {policy} | {statistics.mean(d['em']):.4f} | "
                         f"{statistics.mean(d['f1']):.4f} | {statistics.mean(d['judge']):.4f} | {len(d['em'])} |")

    md_lines += ["", f"EM=0 but judge=CORRECT (EM undercounting genuinely correct answers): "
                      f"{len(disagreements)}/{len(rows)} ({len(disagreements)/len(rows):.1%})", "",
                 "| Benchmark | Policy | Question | Reference | Prediction |",
                 "|---|---|---|---|---|"]
    for r in disagreements[:20]:
        md_lines.append(f"| {r['benchmark']} | {r['policy']} | {r['question']} | "
                         f"{r['reference']} | {r['prediction']} |")
    if len(disagreements) > 20:
        md_lines.append(f"| ... | ... | ({len(disagreements) - 20} more, see raw output) | | |")

    md = "\n".join(md_lines)
    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)

    stem = Path(args.input).stem
    table_path = out_dir / "tables" / f"week6_judge_scores_{stem}.md"
    raw_path = out_dir / "raw" / f"week6_judge_scores_{stem}.json"
    table_path.write_text(md + f"\n\ntoken usage (includes cache hits at face value): {usage_totals}\n",
                           encoding="utf-8")
    raw_path.write_text(json.dumps(rows, indent=2))
    print(f"\ntoken usage (includes cache hits at face value): {usage_totals}")

    print(md)
    print(f"\nwritten -> {table_path}")


if __name__ == "__main__":
    main()
