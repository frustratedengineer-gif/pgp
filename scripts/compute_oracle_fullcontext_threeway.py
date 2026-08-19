#!/usr/bin/env python
"""
Final-consistency-pass fix: paper/draft.md Section 6.14 and README.md cite
"N=22 in the matched oracle/full_context/no_forget subset" for the
three-way LoCoMo comparison, but that number was computed ad hoc during
drafting and never written to a committed table -- a real gap against
this project's own stated standard that every number traces to a file
under results/tables/. This recomputes and commits it.

Three-way match on (conversation_id, question): only LoCoMo questions
where `oracle`, `full_context`, AND `no_forget` all have a scored row --
`oracle` requires evidence coverage (not every question has it),
`full_context` was only run for a capped 3-questions-per-conversation
sample (see scripts/eval_oracle_fullcontext.py), so the intersection is
smaller than any single policy's own N.

    python scripts/compute_oracle_fullcontext_threeway.py
"""
import json
import statistics
from pathlib import Path

from memorylife.evaluation.downstream_significance import bootstrap_paired_mean_diff


def main():
    oracle_fc = json.loads(Path("results/raw/week6_oracle_fullcontext_raw.json").read_text(encoding="utf-8"))
    pilot = json.loads(Path("results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json").read_text(encoding="utf-8"))

    oracle_by_q = {(r["conversation_id"], r["question"]): r for r in oracle_fc
                   if r["benchmark"] == "locomo" and r["policy"] == "oracle"}
    fc_by_q = {(r["conversation_id"], r["question"]): r for r in oracle_fc
               if r["benchmark"] == "locomo" and r["policy"] == "full_context"}
    nf_by_q = {(r["conversation_id"], r["question"]): r for r in pilot
               if r["benchmark"] == "locomo" and r["policy"] == "no_forget"}

    common = sorted(set(oracle_by_q) & set(fc_by_q) & set(nf_by_q))

    md_lines = [
        "# Oracle vs. full_context vs. no_forget: three-way matched comparison (LoCoMo)",
        "",
        "Follow-up to week6_oracle_fullcontext.md / week6_oracle_significance.md -- those tables "
        "report each policy's own full sample (oracle N=107, full_context N=30, no_forget N=120, "
        "all different because `oracle` requires evidence coverage and `full_context` was only run "
        "for a capped 3-questions-per-conversation sample). This is the SAME 3 policies restricted "
        "to the exact questions all three have a scored row for, so mean/significance comparisons "
        "here are genuinely paired, not just three separate marginal means.",
        "",
        f"**N = {len(common)}** questions where oracle, full_context, and no_forget were all scored.",
        "",
        "| Policy | N | Mean EM | Mean F1 |",
        "|---|---|---|---|",
    ]
    for name, by_q in (("oracle", oracle_by_q), ("full_context", fc_by_q), ("no_forget", nf_by_q)):
        rows = [by_q[k] for k in common]
        md_lines.append(f"| {name} | {len(rows)} | {statistics.mean(r['em'] for r in rows):.4f} | "
                         f"{statistics.mean(r['f1'] for r in rows):.4f} |")

    md_lines += ["", "## Pairwise bootstrap significance (n_boot=10000, seed=42)", "",
                 "| Comparison (a vs b) | EM diff (a-b), 95% CI | p (a<=b) | F1 diff (a-b), 95% CI | p (a<=b) |",
                 "|---|---|---|---|---|"]
    pairs = [("oracle", oracle_by_q, "full_context", fc_by_q),
             ("full_context", fc_by_q, "no_forget", nf_by_q),
             ("oracle", oracle_by_q, "no_forget", nf_by_q)]
    for name_a, dict_a, name_b, dict_b in pairs:
        em_a = [dict_a[k]["em"] for k in common]
        em_b = [dict_b[k]["em"] for k in common]
        f1_a = [dict_a[k]["f1"] for k in common]
        f1_b = [dict_b[k]["f1"] for k in common]
        r_em = bootstrap_paired_mean_diff(em_a, em_b, n_boot=10000, seed=42)
        r_f1 = bootstrap_paired_mean_diff(f1_a, f1_b, n_boot=10000, seed=42)
        md_lines.append(f"| {name_a} vs {name_b} | {r_em['mean_diff']:+.4f} "
                         f"[{r_em['ci_low']:+.4f}, {r_em['ci_high']:+.4f}] | p={r_em['p_value_one_sided']:.4f} | "
                         f"{r_f1['mean_diff']:+.4f} [{r_f1['ci_low']:+.4f}, {r_f1['ci_high']:+.4f}] | "
                         f"p={r_f1['p_value_one_sided']:.4f} |")

    md = "\n".join(md_lines)
    out_path = Path("results/tables/week6_oracle_fullcontext_threeway.md")
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
