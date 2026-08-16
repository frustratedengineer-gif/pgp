#!/usr/bin/env python
"""Links the raw LoCoMo/LongMemEval benchmark files to our processed data
and reports evidence coverage -- a sanity check before running
scripts/run_downstream_qa_eval.py, and the real ceiling on achievable
downstream QA accuracy (evidence never extracted can't be answered by any
forgetting policy, so this isn't a policy failure if it's below 100%).

    python scripts/build_benchmark.py
"""
import argparse

from memorylife.data.build_benchmark import evidence_coverage, load_all_processed, load_locomo_qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--feat-dir", default="artifacts/features")
    ap.add_argument("--locomo-path", default="data/raw/locomo10.json")
    args = ap.parse_args()

    processed = load_all_processed(args.data_dir, args.emb_dir, args.feat_dir)
    locomo_qa = load_locomo_qa(args.locomo_path)
    coverage = evidence_coverage(locomo_qa, processed)

    print(f"LoCoMo conversations linked: {len(locomo_qa)}")
    for sample_id, cov in sorted(coverage.items()):
        split = processed["conversation_to_split"].get(sample_id, "?")
        n_memories = len(processed["conversation_to_memory_ids"].get(sample_id, []))
        print(f"  {sample_id} ({split}, {n_memories} memories): evidence coverage = {cov:.1%}")
    mean_cov = sum(coverage.values()) / len(coverage) if coverage else 0.0
    print(f"\nmean evidence coverage across all LoCoMo conversations: {mean_cov:.1%}")


if __name__ == "__main__":
    main()
