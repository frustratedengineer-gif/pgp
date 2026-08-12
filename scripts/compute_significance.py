#!/usr/bin/env python
"""Week-4: is "our model beats the baselines" actually significant, or could
the C-index gap be sampling noise from the ~900-record test set?

Bootstraps a paired CI + one-sided p-value for (our_model C-index - baseline
C-index) on each split, for every baseline that has scores available. See
src/memorylife/evaluation/significance.py for the method.

    python scripts/compute_significance.py --splits val test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import METHOD_LABELS, OPTIONAL_METHODS, REQUIRED_METHODS, baseline_scores, our_model_scores  # noqa: E402

from memorylife.data.datasets import load_split  # noqa: E402
from memorylife.encoders.cache import ensure_embeddings  # noqa: E402
from memorylife.evaluation.significance import bootstrap_c_index, bootstrap_paired_diff  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--scores-dir", default="artifacts/scores")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ensure_embeddings(args.data_dir, args.emb_dir, args.splits, device=args.device)

    rows = []
    for split_name in args.splits:
        split = load_split(args.data_dir, args.emb_dir, split_name)
        durations, events = split["durations"], split["events"]

        our_scores_dict = our_model_scores(args.model, split)
        our_scores = [our_scores_dict[mid] for mid in split["ids"]]
        our_ci = bootstrap_c_index(durations, our_scores, events, n_boot=args.n_boot)
        rows.append({
            "split": split_name, "method": METHOD_LABELS["our_model"],
            "c_index_ci": f"[{our_ci['ci_low']:.4f}, {our_ci['ci_high']:.4f}]",
            "vs_our_model": "-", "p_value_one_sided": "-",
        })

        for method in (*REQUIRED_METHODS, *OPTIONAL_METHODS):
            path = Path(args.scores_dir) / f"{method}_{split_name}.json"
            if not path.exists():
                continue
            base_dict = baseline_scores(args.scores_dir, method, split_name)
            base_scores = [base_dict[mid] for mid in split["ids"]]
            base_ci = bootstrap_c_index(durations, base_scores, events, n_boot=args.n_boot)
            diff = bootstrap_paired_diff(durations, our_scores, base_scores, events, n_boot=args.n_boot)
            rows.append({
                "split": split_name, "method": METHOD_LABELS[method],
                "c_index_ci": f"[{base_ci['ci_low']:.4f}, {base_ci['ci_high']:.4f}]",
                "vs_our_model": f"{diff['mean_diff']:+.4f} [{diff['ci_low']:+.4f}, {diff['ci_high']:+.4f}]",
                "p_value_one_sided": f"{diff['p_value_one_sided']:.4f}",
            })

    out_dir = Path(args.out_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "| Split | Method | C-index 95% CI | Our model - method (95% CI) | p (one-sided, ours <= method) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['split']} | {r['method']} | {r['c_index_ci']} | {r['vs_our_model']} | {r['p_value_one_sided']} |"
        )
    (out_dir / "week4_significance.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))
    print(f"\nwritten -> {out_dir / 'week4_significance.md'}")


if __name__ == "__main__":
    main()
