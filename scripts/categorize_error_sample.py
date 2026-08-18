#!/usr/bin/env python
"""
Reviewer gap (from comparing against REMem, ICLR 2026, which samples 100
errors and buckets them into named failure categories): every claim in
Weeks 3-6 so far is a rate or a handful of hand-picked qualitative
examples, never a systematic breakdown of HOW the wrong answers fail.

Draws the SAME reproducible 100-question sample (seed=42, from the
judge-scored pilot's 512 wrong -- judge=0 or, if unjudged, em=0 --
predictions) and applies a fixed category label per row. The category
labels below were assigned by manual read-through (by the AI assistant
doing this work, not an independent second human rater -- see the
"Methodology" note in the output table; this is a real limitation,
flagged rather than presented as a REMem-equivalent human study).

Categories:
  REFUSAL              -- answered "I don't have that information" (or a
                           close variant) on a question that DOES have a
                           reference answer, i.e. a false refusal.
  WRONG_VALUE           -- a confident, specific, but incorrect fact
                           (wrong date/name/number/entity, or an
                           unrelated hallucinated detail).
  INCOMPLETE            -- captured only part of a multi-item reference
                           answer (e.g. 1 of 3 named items).
  DATE_OFF_BY_ONE        -- a date/timezone/relative-date conversion
                           error landing within ~1 day/unit of the
                           reference, distinct from a wholesale wrong
                           value.
  REASONING_ERROR        -- dropped or mishandled a comparative/causal
                           relationship in the question itself (e.g. a
                           "both X and Y" question, or "why" question
                           answered with the wrong causal link).
  JUDGE_ERROR_CANDIDATE  -- the prediction looks substantively correct on
                           inspection; likely an LLM-judge scoring
                           mistake, not a real system failure (same
                           caveat pattern as week6_qualitative_examples.md).

    python scripts/categorize_error_sample.py
"""
import json
import random
from pathlib import Path

CATEGORIES = [
    "WRONG_VALUE", "REFUSAL", "REFUSAL", "REFUSAL", "REFUSAL", "WRONG_VALUE", "DATE_OFF_BY_ONE",
    "REFUSAL", "INCOMPLETE", "REFUSAL", "WRONG_VALUE", "WRONG_VALUE", "WRONG_VALUE", "REFUSAL",
    "REFUSAL", "WRONG_VALUE", "JUDGE_ERROR_CANDIDATE", "WRONG_VALUE", "REFUSAL", "WRONG_VALUE",
    "REFUSAL", "DATE_OFF_BY_ONE", "REFUSAL", "REFUSAL", "JUDGE_ERROR_CANDIDATE", "INCOMPLETE",
    "REFUSAL", "REFUSAL", "REFUSAL", "REFUSAL", "INCOMPLETE", "DATE_OFF_BY_ONE", "REFUSAL",
    "REASONING_ERROR", "DATE_OFF_BY_ONE", "WRONG_VALUE", "REFUSAL", "REFUSAL", "REFUSAL", "REFUSAL",
    "INCOMPLETE", "INCOMPLETE", "WRONG_VALUE", "INCOMPLETE", "REFUSAL", "REFUSAL", "INCOMPLETE",
    "WRONG_VALUE", "REFUSAL", "WRONG_VALUE", "WRONG_VALUE", "WRONG_VALUE", "REFUSAL", "REFUSAL",
    "REFUSAL", "INCOMPLETE", "REFUSAL", "REFUSAL", "WRONG_VALUE", "WRONG_VALUE", "REFUSAL",
    "REFUSAL", "REFUSAL", "REFUSAL", "REFUSAL", "DATE_OFF_BY_ONE", "REFUSAL", "WRONG_VALUE",
    "WRONG_VALUE", "REFUSAL", "REFUSAL", "REFUSAL", "REFUSAL", "INCOMPLETE", "REFUSAL", "REFUSAL",
    "REFUSAL", "INCOMPLETE", "INCOMPLETE", "REFUSAL", "REFUSAL", "REFUSAL", "JUDGE_ERROR_CANDIDATE",
    "REASONING_ERROR", "REFUSAL", "REFUSAL", "REFUSAL", "REFUSAL", "INCOMPLETE", "REFUSAL",
    "INCOMPLETE", "REFUSAL", "WRONG_VALUE", "WRONG_VALUE", "REFUSAL", "REFUSAL", "INCOMPLETE",
    "INCOMPLETE", "WRONG_VALUE", "REFUSAL",
]

CATEGORY_DESCRIPTIONS = {
    "REFUSAL": "False refusal -- answered \"I don't have that information\" on a question that IS answerable",
    "WRONG_VALUE": "Confident but incorrect specific fact, or an unrelated hallucinated detail",
    "INCOMPLETE": "Captured only part of a multi-item reference answer",
    "DATE_OFF_BY_ONE": "Date/relative-date conversion landed within ~1 day/unit of the reference",
    "REASONING_ERROR": "Dropped or mishandled a comparative/causal relationship in the question",
    "JUDGE_ERROR_CANDIDATE": "Prediction looks substantively correct; likely an LLM-judge scoring mistake",
}


def main():
    data = json.loads(Path("results/raw/week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json")
                       .read_text(encoding="utf-8"))
    wrong = [r for r in data if r.get("judge", r["em"]) == 0.0]
    rng = random.Random(42)
    sample = rng.sample(wrong, 100)
    assert len(CATEGORIES) == len(sample), f"{len(CATEGORIES)} categories for {len(sample)} sampled rows"

    labeled = []
    for row, category in zip(sample, CATEGORIES):
        labeled.append({"benchmark": row["benchmark"], "policy": row["policy"], "question": row["question"],
                         "reference": str(row["reference"]), "prediction": row["prediction"], "category": category})

    Path("results/raw").mkdir(parents=True, exist_ok=True)
    Path("results/raw/week6_error_taxonomy_sample.json").write_text(json.dumps(labeled, indent=2))

    from collections import Counter
    counts = Counter(r["category"] for r in labeled)
    md_lines = [
        "# Error-mode taxonomy: 100 sampled wrong predictions",
        "",
        "Reviewer gap (comparing against REMem, ICLR 2026, Section 6.4's error analysis): every "
        "claim elsewhere in Week 6 is a rate or a handful of hand-picked qualitative examples, never "
        "a systematic breakdown of HOW the wrong answers fail. This samples 100 wrong (judge=0, or "
        "em=0 where unjudged) predictions -- reproducibly, seed=42, from the same 512-wrong pool "
        "pooled across all 5 policies and both benchmarks in "
        "`week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json` -- and assigns each one "
        "a category.",
        "",
        "**Methodology limitation, stated directly**: unlike REMem's human error analysis, this "
        "categorization was done by single-pass manual read-through by the AI assistant doing this "
        "work, not an independently human-verified second rater -- no inter-rater agreement exists. "
        "Treat category boundaries (especially REFUSAL vs. WRONG_VALUE on borderline partial answers) "
        "as a considered judgment call, not a ground-truth label.",
        "",
        "| Category | Count | % | What it means |",
        "|---|---|---|---|",
    ]
    for cat, desc in CATEGORY_DESCRIPTIONS.items():
        c = counts.get(cat, 0)
        md_lines.append(f"| {cat} | {c} | {c}% | {desc} |")

    md_lines += [
        "",
        "## Reading this",
        "",
        f"**{counts['REFUSAL']}% of all sampled failures are false refusals**, not wrong guesses -- "
        "the single largest category by a wide margin. This is the same phenomenon "
        "`week6_refusal_eval.md` measures directly and with significance testing (refusal precision "
        f"varies significantly by policy); this taxonomy shows it's not a minor edge case, it's the "
        f"dominant failure mode across ALL policies pooled. Genuine wrong-value hallucinations "
        f"({counts['WRONG_VALUE']}%) and incomplete multi-item answers ({counts['INCOMPLETE']}%) are "
        f"real but secondary. Date-arithmetic near-misses ({counts['DATE_OFF_BY_ONE']}%) and dropped "
        f"comparative/causal reasoning ({counts['REASONING_ERROR']}%) are minority failure modes -- "
        "the system is not primarily failing at temporal arithmetic, contrary to what a reader might "
        "assume given how much this project's write-ups discuss TTL/temporal calibration. "
        f"{counts['JUDGE_ERROR_CANDIDATE']}% look like judge scoring mistakes on inspection, "
        "consistent with (not larger than) the caveat already raised in "
        "`week6_qualitative_examples.md`.",
        "",
    ]

    md = "\n".join(md_lines)
    Path("results/tables/week6_error_taxonomy.md").write_text(md, encoding="utf-8")
    print(md)
    print("\nwritten -> results/tables/week6_error_taxonomy.md")


if __name__ == "__main__":
    main()
