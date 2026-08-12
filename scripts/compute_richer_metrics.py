#!/usr/bin/env python
"""Week-4: metrics beyond C-index -- time-dependent AUC for every method,
Brier score / IBS for our_model (a real survival curve) and for every
scalar-TTL baseline (via a degenerate step-function curve -- see
src/memorylife/evaluation/richer_metrics.py for why that's a legitimate,
clearly-labeled way to Brier-score a point forecast).

    python scripts/compute_richer_metrics.py --splits val test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import METHOD_LABELS, OPTIONAL_METHODS, REQUIRED_METHODS, baseline_scores  # noqa: E402

import pandas as pd  # noqa: E402

from memorylife.data.datasets import load_split  # noqa: E402
from memorylife.encoders.cache import ensure_embeddings  # noqa: E402
from memorylife.evaluation.richer_metrics import (  # noqa: E402
    brier_and_ibs, cox_surv_probs, eval_times_for, step_function_surv_probs, time_dependent_auc,
)
from memorylife.models.checkpoint import load_survival_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--scores-dir", default="artifacts/scores")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--n-times", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ensure_embeddings(args.data_dir, args.emb_dir, [*args.splits, "train"], device=args.device)
    train = load_split(args.data_dir, args.emb_dir, "train")
    x_train = train["embeddings"].astype("float32")

    model = load_survival_model(args.model, x_train.shape[1])
    # checkpoints don't persist baseline hazards -- recompute from the same
    # training split used to fit the model (fast: same cost as one epoch)
    model.compute_baseline_hazards(input=x_train, target=(train["durations"], train["events"]))

    rows = []
    for split_name in args.splits:
        split = load_split(args.data_dir, args.emb_dir, split_name)
        durations, events = split["durations"], split["events"]
        eval_times = eval_times_for(durations, n_times=args.n_times)

        risk = model.predict(split["embeddings"].astype("float32")).flatten()
        our_scores = -risk
        auc = time_dependent_auc(train["durations"], train["events"], durations, events, our_scores, eval_times)
        surv_probs = cox_surv_probs(model, split["embeddings"], eval_times)
        bs = brier_and_ibs(train["durations"], train["events"], durations, events, surv_probs, eval_times)
        rows.append({"split": split_name, "method": METHOD_LABELS["our_model"],
                      "mean_auc": round(auc["mean_auc"], 4), "ibs": round(bs["ibs"], 4), "n": len(durations)})

        for method in (*REQUIRED_METHODS, *OPTIONAL_METHODS):
            path = Path(args.scores_dir) / f"{method}_{split_name}.json"
            if not path.exists():
                continue
            scores_dict = baseline_scores(args.scores_dir, method, split_name)
            scores = [scores_dict[mid] for mid in split["ids"]]  # already "predicted days"
            auc = time_dependent_auc(train["durations"], train["events"], durations, events, scores, eval_times)
            surv_probs = step_function_surv_probs(scores, eval_times)
            bs = brier_and_ibs(train["durations"], train["events"], durations, events, surv_probs, eval_times)
            rows.append({"split": split_name, "method": METHOD_LABELS[method],
                          "mean_auc": round(auc["mean_auc"], 4), "ibs": round(bs["ibs"], 4), "n": len(durations)})

    df = pd.DataFrame(rows).sort_values(["split", "ibs"])
    out_dir = Path(args.out_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "week4_richer_metrics.csv", index=False)

    md_lines = [
        "| Split | Method | Mean time-dependent AUC | Integrated Brier Score (lower better) | N |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        md_lines.append(f"| {r['split']} | {r['method']} | {r['mean_auc']:.4f} | {r['ibs']:.4f} | {r['n']} |")
    (out_dir / "week4_richer_metrics.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))
    print("\nNote: Brier/IBS for non-our_model methods use a degenerate step-function survival "
          "curve from their scalar predicted-days output (see richer_metrics.py docstring) -- "
          "they do not produce a calibrated probability curve the way our_model does.")


if __name__ == "__main__":
    main()
