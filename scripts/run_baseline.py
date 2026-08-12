#!/usr/bin/env python
"""Run one Week-3 baseline, writing artifacts/scores/{method}_{split}.json
for each split, in the same score format/direction convention scripts/
evaluate.py expects (see memorylife.evaluation.survival_metrics).

    python scripts/run_baseline.py --method heuristic_ttl
    python scripts/run_baseline.py --method bucket_classifier
    python scripts/run_baseline.py --method llm_prompted_ttl --llm-model qwen2.5:7b-instruct
    OPENROUTER_API_KEY=... python scripts/run_baseline.py --method chatgpt_prompted_ttl --splits val test
    OPENROUTER_API_KEY=... python scripts/run_baseline.py --method gemini_prompted_ttl --splits val test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import (  # noqa: E402
    bucket_classifier, chatgpt_prompted_ttl, gemini_prompted_ttl,
    heuristic_ttl, llm_prompted_ttl,
)

REGISTRY = {
    "heuristic_ttl": heuristic_ttl.run,
    "bucket_classifier": bucket_classifier.run,
    "llm_prompted_ttl": llm_prompted_ttl.run,
    "chatgpt_prompted_ttl": chatgpt_prompted_ttl.run,
    "gemini_prompted_ttl": gemini_prompted_ttl.run,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--out-dir", default="artifacts/scores")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    ap.add_argument("--llm-url", default="http://127.0.0.1:11434/api/generate")
    ap.add_argument("--llm-workers", type=int, default=6)
    ap.add_argument("--device", default="cuda", help="used by bucket_classifier's embedding step")
    ap.add_argument("--seed", type=int, default=42, help="used by bucket_classifier's LogisticRegression")
    ap.add_argument("--chatgpt-model", default="openai/gpt-4o",
                     help="OpenRouter model slug, see https://openrouter.ai/models")
    ap.add_argument("--gemini-model", default="google/gemini-2.5-pro",
                     help="OpenRouter model slug, see https://openrouter.ai/models")
    args = ap.parse_args()

    REGISTRY[args.method](args)


if __name__ == "__main__":
    main()
