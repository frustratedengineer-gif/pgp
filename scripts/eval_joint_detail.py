#!/usr/bin/env python
"""Per-class detail for the Week-5 joint model's Action head, and
precision/recall/F1/AUC for the Future-utility head -- aggregate accuracy
(scripts/train_joint.py's printed summary) hides whether a class-imbalanced
head is actually working per-class or just predicting the majority class
everywhere. Runs over every experiments/joint/<fusion>_seed<N>/ checkpoint
already produced by the seed sweep, no retraining.

    python scripts/eval_joint_detail.py --splits val test
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_joint import build_supervision  # noqa: E402

from memorylife.data.datasets import load_split
from memorylife.features.pipeline import load_features
from memorylife.heads.action import ACTION_LABELS
from memorylife.models.checkpoint import load_joint_model


def evaluate_checkpoint(model_path, data_dir, emb_dir, feat_dir, splits, device):
    model = load_joint_model(model_path)
    model.to(device).eval()

    rows = []
    for split_name in splits:
        split = load_split(data_dir, emb_dir, split_name)
        feats = load_features(feat_dir, split_name)
        action_labels, utility_labels, utility_mask = build_supervision(split["records"])

        with torch.no_grad():
            emb_t = torch.tensor(split["embeddings"], dtype=torch.float32, device=device)
            feat_t = torch.tensor(feats, dtype=torch.float32, device=device)
            out = model(emb_t, feat_t)
            action_pred = out["action_logits"].argmax(dim=-1).cpu().numpy()
            utility_prob = torch.sigmoid(out["utility_logit"]).cpu().numpy()

        report = classification_report(action_labels, action_pred, labels=list(range(len(ACTION_LABELS))),
                                        target_names=ACTION_LABELS, output_dict=True, zero_division=0)

        utility_metrics = {}
        if utility_mask.sum() > 0 and len(set(utility_labels[utility_mask])) > 1:
            y_true = utility_labels[utility_mask].astype(int)  # cast off float32 -- otherwise
            y_prob = utility_prob[utility_mask]                # classification_report's dict keys
            y_pred = (y_prob > 0.5).astype(int)                # come out as "0.0"/"1.0", not "0"/"1"
            u_report = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
            utility_metrics = {
                "auc": float(roc_auc_score(y_true, y_prob)),
                "precision_pos": float(u_report["1"]["precision"]),
                "recall_pos": float(u_report["1"]["recall"]),
                "n_labeled": int(utility_mask.sum()),
            }

        rows.append({"split": split_name, "action_report": report, "utility_metrics": utility_metrics})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--feat-dir", default="artifacts/features")
    ap.add_argument("--joint-dir", default="experiments/joint")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/tables/week5_action_utility_detail.json")
    args = ap.parse_args()

    all_results = {}
    for run_dir in sorted(Path(args.joint_dir).iterdir()):
        model_path = run_dir / "model.pt"
        if not model_path.exists():
            continue
        print(f"== {run_dir.name} ==")
        rows = evaluate_checkpoint(model_path, args.data_dir, args.emb_dir, args.feat_dir, args.splits, args.device)
        all_results[run_dir.name] = rows
        for row in rows:
            print(f"  {row['split']}:")
            for label in ACTION_LABELS:
                m = row["action_report"][label]
                print(f"    action={label:8s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
                      f"f1={m['f1-score']:.3f} support={int(m['support'])}")
            if row["utility_metrics"]:
                um = row["utility_metrics"]
                print(f"    utility: auc={um['auc']:.3f} precision_pos={um['precision_pos']:.3f} "
                      f"recall_pos={um['recall_pos']:.3f} n_labeled={um['n_labeled']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
