# Error-mode taxonomy: 100 sampled wrong predictions

Reviewer gap (comparing against REMem, ICLR 2026, Section 6.4's error analysis): every claim elsewhere in Week 6 is a rate or a handful of hand-picked qualitative examples, never a systematic breakdown of HOW the wrong answers fail. This samples 100 wrong (judge=0, or em=0 where unjudged) predictions -- reproducibly, seed=42, from the same 512-wrong pool pooled across all 5 policies and both benchmarks in `week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json` -- and assigns each one a category.

**Methodology limitation, stated directly**: unlike REMem's human error analysis, this categorization was done by single-pass manual read-through by the AI assistant doing this work, not an independently human-verified second rater -- no inter-rater agreement exists. Treat category boundaries (especially REFUSAL vs. WRONG_VALUE on borderline partial answers) as a considered judgment call, not a ground-truth label.

| Category | Count | % | What it means |
|---|---|---|---|
| REFUSAL | 54 | 54% | False refusal -- answered "I don't have that information" on a question that IS answerable |
| WRONG_VALUE | 21 | 21% | Confident but incorrect specific fact, or an unrelated hallucinated detail |
| INCOMPLETE | 15 | 15% | Captured only part of a multi-item reference answer |
| DATE_OFF_BY_ONE | 5 | 5% | Date/relative-date conversion landed within ~1 day/unit of the reference |
| REASONING_ERROR | 2 | 2% | Dropped or mishandled a comparative/causal relationship in the question |
| JUDGE_ERROR_CANDIDATE | 3 | 3% | Prediction looks substantively correct; likely an LLM-judge scoring mistake |

## Reading this

**54% of all sampled failures are false refusals**, not wrong guesses -- the single largest category by a wide margin. This is the same phenomenon `week6_refusal_eval.md` measures directly and with significance testing (refusal precision varies significantly by policy); this taxonomy shows it's not a minor edge case, it's the dominant failure mode across ALL policies pooled. Genuine wrong-value hallucinations (21%) and incomplete multi-item answers (15%) are real but secondary. Date-arithmetic near-misses (5%) and dropped comparative/causal reasoning (2%) are minority failure modes -- the system is not primarily failing at temporal arithmetic, contrary to what a reader might assume given how much this project's write-ups discuss TTL/temporal calibration. 3% look like judge scoring mistakes on inspection, consistent with (not larger than) the caveat already raised in `week6_qualitative_examples.md`.
