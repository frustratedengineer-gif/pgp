#!/usr/bin/env python
"""
Reviewer gap (comparing against REMem, ICLR 2026, whose every results
table is sliced by question type -- single-hop/multi-hop/temporal/open-
domain for LoCoMo, question_type for others): every Week-6 downstream QA
number so far is one aggregate EM/F1 per policy. This asks whether the
Fix #2 (`ours_utility`) improvement is uniform across question types or
concentrated in a few -- free, reuses already-collected predictions and
judge scores from the q0.2 ranked pilot, no new LLM calls.

LoCoMo's `category` field (verified against data/raw/locomo10.json, and
cross-checked against REMem's own published per-category N -- our counts
match theirs exactly: category 1=Multi-Hop (N=282), 2=Single-Hop (321),
3=Temporal Reasoning (96), 4=Open-Domain (841), 5=Adversarial (446,
excluded from this table, see week6_refusal_eval.md instead).

LongMemEval's own `question_type` field (multi-session, temporal-
reasoning, knowledge-update, single-session-user/assistant/preference) --
official categories, not inferred.

    python scripts/breakdown_by_question_type.py
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

LOCOMO_CATEGORY_NAMES = {1: "Multi-Hop", 2: "Single-Hop", 3: "Temporal", 4: "Open-Domain"}
POLICIES = ("no_forget", "fifo", "lru", "ours", "ours_utility")


def main():
    locomo = json.loads(Path("data/raw/locomo10.json").read_text(encoding="utf-8"))
    q_to_cat: dict[tuple, str] = {}
    for c in locomo:
        for q in c["qa"]:
            cat_num = q.get("category")
            if cat_num in LOCOMO_CATEGORY_NAMES:
                q_to_cat[(c["sample_id"], q["question"])] = LOCOMO_CATEGORY_NAMES[cat_num]

    lme = json.loads(Path("data/raw/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    lme_q_to_type = {d["question"]: d["question_type"] for d in lme}

    predictions = json.loads(Path("results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json")
                              .read_text(encoding="utf-8"))
    judged = json.loads(Path("results/raw/week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json")
                         .read_text(encoding="utf-8"))
    judge_by_key = {(r["benchmark"], r["conversation_id"], r["policy"], r["question"]): r.get("judge")
                     for r in judged}

    # {(benchmark, category, policy): {"em": [...], "f1": [...], "judge": [...]}}
    buckets: dict[tuple, dict] = defaultdict(lambda: {"em": [], "f1": [], "judge": []})
    for r in predictions:
        if r["benchmark"] == "locomo":
            category = q_to_cat.get((r["conversation_id"], r["question"]))
        else:
            category = lme_q_to_type.get(r["question"])
        if category is None:
            continue
        key = (r["benchmark"], category, r["policy"])
        buckets[key]["em"].append(r["em"])
        buckets[key]["f1"].append(r["f1"])
        judge = judge_by_key.get((r["benchmark"], r["conversation_id"], r["policy"], r["question"]))
        if judge is not None:
            buckets[key]["judge"].append(judge)

    md_lines = [
        "# Downstream QA, broken down by question type (reviewer gap)",
        "",
        "Same q0.2 ranked-pilot predictions as week6_downstream_qa_q0.2_ranked_pilot.md / "
        "week6_judge_scores_*.md, just sliced by question type instead of reported as one aggregate "
        "per policy -- free, no new LLM calls. LoCoMo categories verified against "
        "data/raw/locomo10.json and cross-checked against REMem's own published per-category N "
        "(exact match: Multi-Hop 282, Single-Hop 321, Temporal 96, Open-Domain 841, Adversarial 446).",
        "",
        "**Small-N caveat, stated directly**: the pilot's 120 LoCoMo questions were never designed to "
        "be a category-stratified sample (12 answerable QA pairs per conversation, in file order), so "
        "per-category N is uneven and Open-Domain (N=2) and most LongMemEval categories (N<=8) are too "
        "small to draw a reliable per-category conclusion from alone -- reported for transparency, not "
        "as a confirmed per-category claim.",
        "",
    ]

    for benchmark in ("locomo", "longmemeval"):
        categories = sorted({cat for (b, cat, _p) in buckets if b == benchmark})
        md_lines += [f"## {benchmark}", "", "| Category | Policy | N | Mean EM | Mean F1 | Mean Judge |",
                     "|---|---|---|---|---|---|"]
        for category in categories:
            for policy in POLICIES:
                d = buckets.get((benchmark, category, policy))
                if not d or not d["em"]:
                    continue
                judge_str = f"{statistics.mean(d['judge']):.4f}" if d["judge"] else "n/a"
                md_lines.append(f"| {category} | {policy} | {len(d['em'])} | "
                                 f"{statistics.mean(d['em']):.4f} | {statistics.mean(d['f1']):.4f} | "
                                 f"{judge_str} |")
        md_lines.append("")

    md = "\n".join(md_lines)
    out_path = Path("results/tables/week6_qa_by_category.md")
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
