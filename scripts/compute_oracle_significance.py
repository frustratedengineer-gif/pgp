#!/usr/bin/env python
"""
Bootstrap significance for the oracle vs. no_forget gap in
week6_oracle_fullcontext.md (reviewer gap follow-up -- same pattern as
compute_downstream_significance.py and compute_refusal_significance.py:
a raw-mean comparison isn't a claim until it's significance-tested).

Paired on the same (conversation_id, question) key across the oracle raw
predictions (scripts/eval_oracle_fullcontext.py) and the no_forget rows
in the original ranked-pilot raw predictions -- only the intersection
(questions covered by BOTH, i.e. oracle could be built) is used.

    python scripts/compute_oracle_significance.py
"""
import json
from pathlib import Path

from memorylife.evaluation.downstream_significance import bootstrap_paired_mean_diff


def main():
    oracle_fc = json.loads(Path("results/raw/week6_oracle_fullcontext_raw.json").read_text(encoding="utf-8"))
    pilot = json.loads(Path("results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json").read_text(encoding="utf-8"))

    md_lines = [
        "# Oracle vs. no_forget: bootstrap significance (follow-up to week6_oracle_fullcontext.md)",
        "",
        "n_boot=10000, seed=42, paired on (conversation_id, question) -- only questions where an oracle "
        "answer could be built (i.e. evidence coverage exists) are included.",
        "",
        "| Benchmark | Comparison (a vs b) | N | EM diff (a-b), 95% CI | p (a<=b) | F1 diff (a-b), 95% CI | p (a<=b) |",
        "|---|---|---|---|---|---|---|",
    ]
    for benchmark in ("locomo", "longmemeval"):
        oracle_by_q = {(r["conversation_id"], r["question"]): r for r in oracle_fc
                       if r["benchmark"] == benchmark and r["policy"] == "oracle"}
        nf_by_q = {(r["conversation_id"], r["question"]): r for r in pilot
                   if r["benchmark"] == benchmark and r["policy"] == "no_forget"}
        common = sorted(set(oracle_by_q) & set(nf_by_q))
        em_a = [oracle_by_q[k]["em"] for k in common]
        em_b = [nf_by_q[k]["em"] for k in common]
        f1_a = [oracle_by_q[k]["f1"] for k in common]
        f1_b = [nf_by_q[k]["f1"] for k in common]
        r_em = bootstrap_paired_mean_diff(em_a, em_b, n_boot=10000, seed=42)
        r_f1 = bootstrap_paired_mean_diff(f1_a, f1_b, n_boot=10000, seed=42)
        md_lines.append(
            f"| {benchmark} | oracle vs no_forget | n={len(common)} | "
            f"{r_em['mean_diff']:+.4f} [{r_em['ci_low']:+.4f}, {r_em['ci_high']:+.4f}] | "
            f"p={r_em['p_value_one_sided']:.4f} | "
            f"{r_f1['mean_diff']:+.4f} [{r_f1['ci_low']:+.4f}, {r_f1['ci_high']:+.4f}] | "
            f"p={r_f1['p_value_one_sided']:.4f} |"
        )

    md = "\n".join(md_lines)
    out_path = Path("results/tables/week6_oracle_significance.md")
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
