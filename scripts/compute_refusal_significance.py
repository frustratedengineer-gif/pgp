#!/usr/bin/env python
"""
Bootstrap significance for the refusal-precision differences in
week6_refusal_eval.md (reviewer gap follow-up -- that table reports raw
precision/recall/F1 with no CI or p-value, the same gap
compute_downstream_significance.py closed for the EM/F1 claims).

Paired: every policy was scored against the SAME 240 questions per
conversation (120 answerable + 120 adversarial), so the same resampled
question indices are reused across policies in each bootstrap replicate.

    python scripts/compute_refusal_significance.py
"""
import json
import statistics
from pathlib import Path

import numpy as np

POLICIES = ("no_forget", "fifo", "lru", "ours", "ours_utility")
COMPARISONS = [("ours_utility", "ours"), ("ours_utility", "fifo"), ("no_forget", "ours"), ("no_forget", "fifo")]


def bootstrap_precision_diff(rows_a: list[dict], rows_b: list[dict], n_boot: int = 10000, seed: int = 42) -> dict:
    """rows_a[i]/rows_b[i] must be the SAME question under policy a/b.
    Precision = TP / (TP + FP) where TP = refused & adversarial,
    FP = refused & NOT adversarial. Resamples replicates where either
    policy has zero refusals (undefined precision) are dropped."""
    rng = np.random.default_rng(seed)
    n = len(rows_a)
    assert n == len(rows_b)
    refusal_a = np.array([r["is_refusal"] for r in rows_a])
    adv_a = np.array([r["is_adversarial"] for r in rows_a])
    refusal_b = np.array([r["is_refusal"] for r in rows_b])
    adv_b = np.array([r["is_adversarial"] for r in rows_b])

    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = []
    for row_idx in idx:
        ra, aa = refusal_a[row_idx], adv_a[row_idx]
        rb, ab = refusal_b[row_idx], adv_b[row_idx]
        n_refusals_a, n_refusals_b = ra.sum(), rb.sum()
        if n_refusals_a == 0 or n_refusals_b == 0:
            continue
        prec_a = (ra & aa).sum() / n_refusals_a
        prec_b = (rb & ab).sum() / n_refusals_b
        diffs.append(prec_a - prec_b)
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()), "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)), "p_value_one_sided": float((diffs <= 0).mean()),
        "n": n, "n_boot": len(diffs),
    }


def main():
    rows = json.loads(Path("results/raw/week6_refusal_eval_raw.json").read_text(encoding="utf-8"))
    by_policy_question: dict[tuple, dict] = {}
    for r in rows:
        by_policy_question.setdefault(r["policy"], {})[(r["conversation_id"], r["question"])] = r

    # all policies share the same question set (built from the same qa_pool per conversation)
    keys = sorted(by_policy_question[POLICIES[0]].keys())

    md_lines = [
        "# Refusal-precision bootstrap significance (follow-up to week6_refusal_eval.md)",
        "",
        "n_boot=10000, seed=42, paired resampling over the shared 240-question set "
        "(120 answerable + 120 adversarial) per policy.",
        "",
        "| Comparison (a vs b) | N | Precision diff (a-b), 95% CI | p (a<=b) |",
        "|---|---|---|---|",
    ]
    for a, b in COMPARISONS:
        rows_a = [by_policy_question[a][k] for k in keys]
        rows_b = [by_policy_question[b][k] for k in keys]
        result = bootstrap_precision_diff(rows_a, rows_b)
        md_lines.append(f"| {a} vs {b} | n={result['n']} | "
                         f"{result['mean_diff']:+.4f} [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}] | "
                         f"p={result['p_value_one_sided']:.3f} |")

    md = "\n".join(md_lines)
    out_path = Path("results/tables/week6_refusal_significance.md")
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
