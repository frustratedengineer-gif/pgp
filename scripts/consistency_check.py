#!/usr/bin/env python
"""Week-4: how consistent is an LLM's TTL prediction across different
wordings of the *same question*? A second axis of "why not just ask an
LLM" beyond raw C-index (see results/tables/week3_results_table.md):
low accuracy is one problem, but a prediction that swings depending on
phrasing is a second, independent problem for a system that has to be
reliable.

Re-queries a fixed subsample of the test split with 3 paraphrases of the
same prompt (see PARAPHRASES below) for each LLM-prompted baseline (local
Qwen2.5-7B, GPT-4o, Gemini), and reports the coefficient of variation
(std/mean) of the predicted-days output per record, averaged across the
sample. Our trained model isn't re-run here: given fixed weights it is
exactly deterministic (CV=0 by construction) -- that determinism is itself
part of the comparison, not an oversight.

    OPENROUTER_API_KEY=... python scripts/consistency_check.py --n-sample 100
"""
import argparse
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines._openrouter_client import PROMPT_TEMPLATE as BASE_PROMPT  # noqa: E402
from baselines._openrouter_client import _query_one as _query_openrouter  # noqa: E402
from baselines.llm_prompted_ttl import _query_one as _query_ollama  # noqa: E402

from memorylife.utils.io import load_jsonl  # noqa: E402

# same semantic ask ("how many days until this is stale?"), reworded --
# tests robustness to phrasing, not a different question
PARAPHRASES = [
    BASE_PROMPT,
    """Imagine you're managing memory for a personal AI assistant. Here's something the user said: "{text}"

How many days do you expect this piece of information to stay relevant and worth remembering before it goes stale? Permanent facts should get a big number (e.g. 3650 for "basically forever"); short-lived facts might only matter for a day or two, others for weeks or months.

Respond with just one integer -- the number of days. Nothing else.""",
    """A personal AI assistant needs to decide how long to keep a memory around. The memory is: "{text}"

Estimate, in days starting today, how long this information will remain accurate and useful before it's outdated. Use ~3650 for things that are essentially permanent, smaller numbers for short-lived facts (a day or two) or medium-lived ones (weeks/months).

Output only a single number of days -- no other text.""",
]


def coefficient_of_variation(values: list[float]) -> float:
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return statistics.pstdev(values) / mean


def run_paraphrase_batch(kind, model, records, template, api_key=None, cache_dir=None,
                          ollama_url=None, workers=4) -> dict[str, float]:
    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        if kind == "ollama":
            futures = [ex.submit(_query_ollama, ollama_url, model, r["memory_id"], r["text"], template)
                       for r in records]
            for fut in futures:
                mid, val = fut.result()
                out[mid] = val
        else:
            futures = [ex.submit(_query_openrouter, api_key, model, r["memory_id"], r["text"],
                                  cache_dir, template) for r in records]
            for fut in futures:
                mid, val, _usage = fut.result()
                out[mid] = val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache-dir", default="artifacts/llm_cache")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    ap.add_argument("--ollama-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--ollama-workers", type=int, default=6)
    ap.add_argument("--chatgpt-model", default="openai/gpt-4o")
    ap.add_argument("--gemini-model", default="google/gemini-2.5-pro")
    ap.add_argument("--remote-workers", type=int, default=4)
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in environment -- export it and re-run.")

    records = load_jsonl(Path(args.data_dir) / f"{args.split}_survival.jsonl")
    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n_sample, len(records)))
    n_calls = len(sample) * len(PARAPHRASES) * 3
    print(f"consistency check: {len(sample)} records x {len(PARAPHRASES)} paraphrases x 3 methods "
          f"= {n_calls} calls (cached where a prior run already covered a (model,prompt) pair)")

    methods = {
        "local_qwen": ("ollama", args.ollama_model, args.ollama_workers),
        "chatgpt_gpt4o": ("openrouter", args.chatgpt_model, args.remote_workers),
        "gemini": ("openrouter", args.gemini_model, args.remote_workers),
    }

    per_record_rows = []
    per_method_preds = {name: {r["memory_id"]: [] for r in sample} for name in methods}

    for method_name, (kind, model, workers) in methods.items():
        cache_dir = Path(args.cache_dir) / f"consistency_{method_name}"
        for p_idx, template in enumerate(PARAPHRASES):
            preds = run_paraphrase_batch(kind, model, sample, template, api_key=api_key,
                                          cache_dir=cache_dir, ollama_url=args.ollama_url, workers=workers)
            for mid, val in preds.items():
                per_method_preds[method_name][mid].append(val)
            print(f"  {method_name} paraphrase {p_idx}: done")

    for method_name in methods:
        cvs = []
        for r in sample:
            preds = per_method_preds[method_name][r["memory_id"]]
            cv = coefficient_of_variation(preds)
            cvs.append(cv)
            per_record_rows.append({
                "method": method_name, "memory_id": r["memory_id"],
                "predictions_days": preds, "mean_days": round(statistics.mean(preds), 2),
                "cv": round(cv, 4),
            })

    out_dir = Path(args.out_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "week4_consistency_raw.json").write_text(json.dumps(per_record_rows, indent=2))

    method_labels = {
        "local_qwen": "LLM-prompted TTL (local Qwen2.5-7B)",
        "chatgpt_gpt4o": "LLM-prompted TTL (ChatGPT / GPT-4o)",
        "gemini": "LLM-prompted TTL (Gemini)",
    }
    summary_rows = []
    for method_name in methods:
        cvs = [row["cv"] for row in per_record_rows if row["method"] == method_name]
        summary_rows.append({
            "method": method_labels[method_name],
            "mean_cv": round(statistics.mean(cvs), 4),
            "median_cv": round(statistics.median(cvs), 4),
            "pct_records_cv_over_0.5": round(100 * sum(c > 0.5 for c in cvs) / len(cvs), 1),
            "n_records": len(cvs),
        })
    summary_rows.sort(key=lambda r: r["mean_cv"])

    md_lines = [
        "| Method | Mean CV | Median CV | % records with CV > 0.5 | N records |",
        "|---|---|---|---|---|",
        "| Our survival model (CoxPH on BGE embeddings) | 0.0000 | 0.0000 | 0.0 | (deterministic given fixed weights, not re-run here) |",
    ]
    for r in summary_rows:
        md_lines.append(f"| {r['method']} | {r['mean_cv']:.4f} | {r['median_cv']:.4f} | "
                         f"{r['pct_records_cv_over_0.5']}% | {r['n_records']} |")
    (out_dir / "week4_consistency.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))
    print(f"\nCV = coefficient of variation (std/mean) of the predicted TTL-in-days across "
          f"{len(PARAPHRASES)} reworded-but-semantically-identical prompts, per record, averaged over "
          f"{len(sample)} records from {args.split}. Higher = less consistent under rewording.")


if __name__ == "__main__":
    main()
