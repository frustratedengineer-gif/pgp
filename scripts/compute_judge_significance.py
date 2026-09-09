#!/usr/bin/env python
"""
Limitation closed (Section 7 / 6.8): the LLM-judge rescoring in Section 6.8
was reported as raw means only ("would need the same bootstrap treatment as
Section 6.6 before being cited as decisive") -- this applies that exact same
paired-bootstrap treatment (see scripts/compute_downstream_significance.py
and src/memorylife/evaluation/downstream_significance.py) to the "judge"
field already present in results/raw/week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json.
No new LLM calls, no new cost -- the judge scores were already computed for
Section 6.8 (725 predictions, GPT-4o judging each one).

    python scripts/compute_judge_significance.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from memorylife.evaluation.downstream_significance import bootstrap_paired_mean_diff

RAW = Path("results/raw/week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json")
OUT = Path("results/tables/week6_judge_significance.md")


def by_policy_question(rows, policy):
    return {(r["benchmark"], r["conversation_id"], r["question"]): r
            for r in rows if r["policy"] == policy}


def paired_judge_arrays(rows, policy_a, policy_b, benchmark):
    a_by_key = by_policy_question(rows, policy_a)
    b_by_key = by_policy_question(rows, policy_b)
    keys = sorted(k for k in (set(a_by_key) & set(b_by_key)) if k[0] == benchmark)
    assert keys, f"no overlapping questions for {policy_a} vs {policy_b} on {benchmark}"
    return [a_by_key[k]["judge"] for k in keys], [b_by_key[k]["judge"] for k in keys]


def main():
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    n_boot, seed = 10000, 42

    header = ["| Benchmark | Comparison (a vs b) | N | Judge-score diff (a-b), 95% CI | p (a<=b) |",
              "|---|---|---|---|---|"]
    lines = list(header)

    results = []
    for benchmark in ("locomo", "longmemeval"):
        for other in ("fifo", "ours", "no_forget", "lru"):
            va, vb = paired_judge_arrays(rows, "ours_utility", other, benchmark)
            res = bootstrap_paired_mean_diff(va, vb, n_boot=n_boot, seed=seed)
            results.append((benchmark, other, res))
            lines.append(
                f"| {benchmark} | ours_utility vs {other} | n={res['n']} | "
                f"{res['mean_diff']:+.4f} [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}] | "
                f"p={res['p_value_one_sided']:.3f} |"
            )

    md = "\n".join([
        "# Bootstrap significance of the LLM-judge metric (closes a Limitations gap)",
        "",
        f"n_boot={n_boot}, seed={seed}, paired same-question resampling -- identical method to",
        "results/tables/week6_downstream_significance.md, applied to the `judge` field instead",
        "of EM/F1. Source: results/raw/week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json",
        "(725 GPT-4o-judged predictions, already computed for Section 6.8 -- no new LLM calls).",
        "",
        *lines,
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
