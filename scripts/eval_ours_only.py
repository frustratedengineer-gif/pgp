#!/usr/bin/env python
"""Evaluate ONLY our_model's C-index (no baselines) -- used by ablation
sweeps (scripts/run_ablations.py), where re-running/requiring the full
baseline score files (scripts/evaluate.py's REQUIRED_METHODS) would be
wasted work; an ablation only cares how our model's own C-index moves.

    python scripts/eval_ours_only.py --model artifacts/survival_model_net.pt --splits val test
"""
import argparse
import json
from pathlib import Path

from memorylife.data.datasets import load_split
from memorylife.encoders.cache import ensure_embeddings
from memorylife.evaluation.survival_metrics import c_index_for_split
from memorylife.models.checkpoint import load_survival_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--hidden1", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=64)
    ap.add_argument("--dropout1", type=float, default=0.2)
    ap.add_argument("--dropout2", type=float, default=0.1)
    ap.add_argument("--emb-model-name", default=None)
    ap.add_argument("--out", default=None, help="write {split: c_index} json here")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ensure_embeddings(args.data_dir, args.emb_dir, args.splits, device=args.device,
                       model_name=args.emb_model_name)
    results = {}
    for split_name in args.splits:
        split = load_split(args.data_dir, args.emb_dir, split_name)
        model = load_survival_model(args.model, split["embeddings"].shape[1], hidden1=args.hidden1,
                                     hidden2=args.hidden2, dropout1=args.dropout1, dropout2=args.dropout2)
        risk = model.predict(split["embeddings"].astype("float32")).flatten()
        scores = {mid: -float(r) for mid, r in zip(split["ids"], risk)}
        c = c_index_for_split(scores, split)
        results[split_name] = round(c, 4)
        print(f"{split_name}: c_index={c:.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
