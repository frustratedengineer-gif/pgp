#!/usr/bin/env python
"""
Bootstrap significance for the Mem0 baseline (reviewer gap #1) vs. this
project's own policies -- same pattern as every other *_significance.py
script in Week 6: a raw-mean comparison isn't a claim until it's tested.

Paired on the same (conversation_id, question) key across
week6_mem0_baseline_raw.json and the original ranked-pilot raw
predictions (both cover the same 120 LoCoMo questions).

    python scripts/compute_mem0_significance.py
"""
import json
from pathlib import Path

from memorylife.evaluation.downstream_significance import bootstrap_paired_mean_diff

POLICIES = ("no_forget", "ours_utility", "lru", "fifo", "ours")


def main():
    mem0 = json.loads(Path("results/raw/week6_mem0_baseline_raw.json").read_text(encoding="utf-8"))
    pilot = json.loads(Path("results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json").read_text(encoding="utf-8"))
    mem0_by_q = {(r["conversation_id"], r["question"]): r for r in mem0}

    md_lines = [
        "# Mem0 baseline: bootstrap significance vs. this project's own policies",
        "",
        "n_boot=10000, seed=42, paired on (conversation_id, question) -- same 120 LoCoMo questions "
        "as week6_mem0_baseline.md and week6_downstream_qa_q0.2_ranked_pilot.md.",
        "",
        "| Comparison (a vs b) | N | EM diff (a-b), 95% CI | p (a<=b) | F1 diff (a-b), 95% CI | p (a<=b) |",
        "|---|---|---|---|---|---|",
    ]
    for policy in POLICIES:
        pol_by_q = {(r["conversation_id"], r["question"]): r for r in pilot
                    if r["benchmark"] == "locomo" and r["policy"] == policy}
        common = sorted(set(mem0_by_q) & set(pol_by_q))
        em_a = [mem0_by_q[k]["em"] for k in common]
        em_b = [pol_by_q[k]["em"] for k in common]
        f1_a = [mem0_by_q[k]["f1"] for k in common]
        f1_b = [pol_by_q[k]["f1"] for k in common]
        r_em = bootstrap_paired_mean_diff(em_a, em_b, n_boot=10000, seed=42)
        r_f1 = bootstrap_paired_mean_diff(f1_a, f1_b, n_boot=10000, seed=42)
        md_lines.append(
            f"| mem0 vs {policy} | n={len(common)} | "
            f"{r_em['mean_diff']:+.4f} [{r_em['ci_low']:+.4f}, {r_em['ci_high']:+.4f}] | "
            f"p={r_em['p_value_one_sided']:.4f} | "
            f"{r_f1['mean_diff']:+.4f} [{r_f1['ci_low']:+.4f}, {r_f1['ci_high']:+.4f}] | "
            f"p={r_f1['p_value_one_sided']:.4f} |"
        )

    md = "\n".join(md_lines)
    out_path = Path("results/tables/week6_mem0_significance.md")
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
