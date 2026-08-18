#!/usr/bin/env python
"""
Reviewer gap (comparing against REMem, ICLR 2026, whose every QA table
reports F1 + BLEU-1 + LLM-judge together): we only had EM/F1. BLEU-1
(memorylife.evaluation.qa_metrics.bleu1) is a pure string metric, so this
recomputes it over predictions ALREADY collected -- free, no new LLM
calls -- and merges it with the existing judge scores into one combined
table, matching REMem's own table shape.

    python scripts/compute_bleu1.py
"""
import json
import statistics
from pathlib import Path

from memorylife.evaluation.qa_metrics import bleu1


def main():
    predictions = json.loads(Path("results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json")
                              .read_text(encoding="utf-8"))
    judged = json.loads(Path("results/raw/week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json")
                         .read_text(encoding="utf-8"))
    judge_by_key = {(r["benchmark"], r["conversation_id"], r["policy"], r["question"]): r.get("judge")
                     for r in judged}

    summary: dict[tuple, dict] = {}
    for r in predictions:
        r["bleu1"] = bleu1(r["prediction"], str(r["reference"]))
        r["judge"] = judge_by_key.get((r["benchmark"], r["conversation_id"], r["policy"], r["question"]))
        key = (r["benchmark"], r["policy"])
        summary.setdefault(key, {"em": [], "f1": [], "bleu1": [], "judge": []})
        summary[key]["em"].append(r["em"])
        summary[key]["f1"].append(r["f1"])
        summary[key]["bleu1"].append(r["bleu1"])
        if r["judge"] is not None:
            summary[key]["judge"].append(r["judge"])

    md_lines = [
        "# Downstream QA: EM + F1 + BLEU-1 + LLM-judge, together (reviewer gap)",
        "",
        "Same q0.2 ranked-pilot predictions as week6_downstream_qa_q0.2_ranked_pilot.md / "
        "week6_judge_scores_*.md, with BLEU-1 (`memorylife.evaluation.qa_metrics.bleu1`, unigram "
        "precision x brevity penalty) computed fresh over the same already-collected predictions -- "
        "free, no new LLM calls. Matches REMem's own table shape (F1 + BLEU-1 + LLM-judge together).",
        "",
        "| Benchmark | Policy | N | Mean EM | Mean F1 | Mean BLEU-1 | Mean Judge |",
        "|---|---|---|---|---|---|---|",
    ]
    for (benchmark, policy), d in sorted(summary.items()):
        judge_str = f"{statistics.mean(d['judge']):.4f}" if d["judge"] else "n/a"
        md_lines.append(f"| {benchmark} | {policy} | {len(d['em'])} | {statistics.mean(d['em']):.4f} | "
                         f"{statistics.mean(d['f1']):.4f} | {statistics.mean(d['bleu1']):.4f} | {judge_str} |")

    md = "\n".join(md_lines)
    out_path = Path("results/tables/week6_downstream_qa_bleu1.md")
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
