#!/usr/bin/env python
"""Week-3 deliverable: our survival model vs. the 3 baselines, by C-index.

Baseline scores must already exist (see scripts/run_baseline.py --method ...
for each of llm_prompted_ttl / bucket_classifier / heuristic_ttl) before
running this.

    python scripts/evaluate.py --data-dir data/processed --emb-dir artifacts/embeddings \
        --model artifacts/survival_model_net.pt --scores-dir artifacts/scores \
        --out-dir results
"""
import argparse
import json
from pathlib import Path

from memorylife.data.datasets import load_split
from memorylife.encoders.cache import ensure_embeddings
from memorylife.evaluation.report import write_results_table
from memorylife.evaluation.survival_metrics import c_index_for_split
from memorylife.models.checkpoint import load_survival_model

METHOD_LABELS = {
    "our_model": "Our survival model (CoxPH on BGE embeddings)",
    "llm_prompted_ttl": "LLM-prompted TTL",
    "bucket_classifier": "Day/week/permanent classifier",
    "heuristic_ttl": "Recency-frequency heuristic",
}


def our_model_scores(model_path: str, split: dict) -> dict[str, float]:
    model = load_survival_model(model_path, split["embeddings"].shape[1])
    risk = model.predict(split["embeddings"].astype("float32")).flatten()
    # negate: higher score must mean "predicted to survive longer" (see
    # evaluation.survival_metrics.score_direction_note)
    return {mid: -float(r) for mid, r in zip(split["ids"], risk)}


def baseline_scores(scores_dir: str, method: str, split_name: str) -> dict[str, float]:
    path = Path(scores_dir) / f"{method}_{split_name}.json"
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--scores-dir", default="artifacts/scores")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ensure_embeddings(args.data_dir, args.emb_dir, args.splits, device=args.device)

    rows = []
    for split_name in args.splits:
        split = load_split(args.data_dir, args.emb_dir, split_name)

        methods = {"our_model": our_model_scores(args.model, split)}
        for method in ("llm_prompted_ttl", "bucket_classifier", "heuristic_ttl"):
            methods[method] = baseline_scores(args.scores_dir, method, split_name)

        for method_key, scores in methods.items():
            c = c_index_for_split(scores, split)
            rows.append({
                "split": split_name,
                "method": METHOD_LABELS[method_key],
                "c_index": round(c, 4),
                "n": len(split["ids"]),
            })

    write_results_table(rows, Path(args.out_dir) / "tables", name="week3_results_table")
    write_results_table(rows, Path(args.out_dir) / "raw", name="week3_results_raw")

    for split_name in args.splits:
        print(f"\n-- {split_name} --")
        for r in sorted([r for r in rows if r["split"] == split_name], key=lambda r: -r["c_index"]):
            print(f"  {r['c_index']:.4f}  {r['method']}")


if __name__ == "__main__":
    main()
