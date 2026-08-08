#!/usr/bin/env python
"""Run one Week-3 baseline, writing artifacts/scores/{method}_{split}.json
for each split, in the same score format/direction convention scripts/
evaluate.py expects (see memorylife.evaluation.survival_metrics).

    python scripts/run_baseline.py --method heuristic_ttl
    python scripts/run_baseline.py --method bucket_classifier
    python scripts/run_baseline.py --method llm_prompted_ttl --llm-model qwen2.5:7b-instruct
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import bucket_classifier, heuristic_ttl, llm_prompted_ttl  # noqa: E402

REGISTRY = {
    "heuristic_ttl": heuristic_ttl.run,
    "bucket_classifier": bucket_classifier.run,
    "llm_prompted_ttl": llm_prompted_ttl.run,
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
    args = ap.parse_args()

    REGISTRY[args.method](args)


if __name__ == "__main__":
    main()
