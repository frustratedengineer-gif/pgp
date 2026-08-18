# Downstream QA, broken down by question type (reviewer gap)

Same q0.2 ranked-pilot predictions as week6_downstream_qa_q0.2_ranked_pilot.md / week6_judge_scores_*.md, just sliced by question type instead of reported as one aggregate per policy -- free, no new LLM calls. LoCoMo categories verified against data/raw/locomo10.json and cross-checked against REMem's own published per-category N (exact match: Multi-Hop 282, Single-Hop 321, Temporal 96, Open-Domain 841, Adversarial 446).

**Small-N caveat, stated directly**: the pilot's 120 LoCoMo questions were never designed to be a category-stratified sample (12 answerable QA pairs per conversation, in file order), so per-category N is uneven and Open-Domain (N=2) and most LongMemEval categories (N<=8) are too small to draw a reliable per-category conclusion from alone -- reported for transparency, not as a confirmed per-category claim.

## locomo

| Category | Policy | N | Mean EM | Mean F1 | Mean Judge |
|---|---|---|---|---|---|
| Multi-Hop | no_forget | 48 | 0.0833 | 0.2808 | 0.1667 |
| Multi-Hop | fifo | 48 | 0.0208 | 0.2248 | 0.1250 |
| Multi-Hop | lru | 48 | 0.0625 | 0.2911 | 0.1875 |
| Multi-Hop | ours | 48 | 0.0208 | 0.2359 | 0.1250 |
| Multi-Hop | ours_utility | 48 | 0.0833 | 0.2773 | 0.1875 |
| Open-Domain | no_forget | 2 | 0.0000 | 0.1053 | 0.0000 |
| Open-Domain | fifo | 2 | 0.0000 | 0.2258 | 0.5000 |
| Open-Domain | lru | 2 | 0.0000 | 0.1053 | 0.0000 |
| Open-Domain | ours | 2 | 0.0000 | 0.1053 | 0.0000 |
| Open-Domain | ours_utility | 2 | 0.0000 | 0.1053 | 0.0000 |
| Single-Hop | no_forget | 54 | 0.0741 | 0.1173 | 0.5185 |
| Single-Hop | fifo | 54 | 0.0185 | 0.0185 | 0.2037 |
| Single-Hop | lru | 54 | 0.0741 | 0.1204 | 0.5000 |
| Single-Hop | ours | 54 | 0.0370 | 0.0723 | 0.2593 |
| Single-Hop | ours_utility | 54 | 0.0741 | 0.1186 | 0.5370 |
| Temporal | no_forget | 16 | 0.1875 | 0.2162 | 0.2500 |
| Temporal | fifo | 16 | 0.1875 | 0.2054 | 0.2500 |
| Temporal | lru | 16 | 0.1875 | 0.2054 | 0.3125 |
| Temporal | ours | 16 | 0.1875 | 0.2162 | 0.2500 |
| Temporal | ours_utility | 16 | 0.1875 | 0.2162 | 0.2500 |

## longmemeval

| Category | Policy | N | Mean EM | Mean F1 | Mean Judge |
|---|---|---|---|---|---|
| knowledge-update | no_forget | 6 | 0.1667 | 0.2778 | 0.3333 |
| knowledge-update | fifo | 6 | 0.1667 | 0.2778 | 0.3333 |
| knowledge-update | lru | 6 | 0.1667 | 0.1667 | 0.1667 |
| knowledge-update | ours | 6 | 0.1667 | 0.2778 | 0.3333 |
| knowledge-update | ours_utility | 6 | 0.1667 | 0.2778 | 0.3333 |
| multi-session | no_forget | 6 | 0.0000 | 0.0415 | 0.3333 |
| multi-session | fifo | 6 | 0.0000 | 0.0415 | 0.3333 |
| multi-session | lru | 6 | 0.0000 | 0.0546 | 0.1667 |
| multi-session | ours | 6 | 0.0000 | 0.0574 | 0.3333 |
| multi-session | ours_utility | 6 | 0.0000 | 0.0574 | 0.3333 |
| single-session-assistant | no_forget | 1 | 0.0000 | 0.0000 | 0.0000 |
| single-session-assistant | fifo | 1 | 0.0000 | 0.0000 | 0.0000 |
| single-session-assistant | lru | 1 | 0.0000 | 0.0000 | 0.0000 |
| single-session-assistant | ours | 1 | 0.0000 | 0.0000 | 0.0000 |
| single-session-assistant | ours_utility | 1 | 0.0000 | 0.0000 | 0.0000 |
| single-session-preference | no_forget | 2 | 0.0000 | 0.0312 | 0.5000 |
| single-session-preference | fifo | 2 | 0.0000 | 0.0312 | 0.5000 |
| single-session-preference | lru | 2 | 0.0000 | 0.0312 | 0.5000 |
| single-session-preference | ours | 2 | 0.0000 | 0.0312 | 0.5000 |
| single-session-preference | ours_utility | 2 | 0.0000 | 0.0312 | 0.5000 |
| single-session-user | no_forget | 2 | 0.5000 | 0.5417 | 1.0000 |
| single-session-user | fifo | 2 | 0.5000 | 0.5417 | 1.0000 |
| single-session-user | lru | 2 | 0.5000 | 0.5417 | 1.0000 |
| single-session-user | ours | 2 | 0.5000 | 0.5417 | 1.0000 |
| single-session-user | ours_utility | 2 | 0.5000 | 0.5417 | 1.0000 |
| temporal-reasoning | no_forget | 8 | 0.0000 | 0.0938 | 0.2500 |
| temporal-reasoning | fifo | 8 | 0.0000 | 0.1042 | 0.3750 |
| temporal-reasoning | lru | 8 | 0.0000 | 0.0938 | 0.2500 |
| temporal-reasoning | ours | 8 | 0.0000 | 0.0938 | 0.2500 |
| temporal-reasoning | ours_utility | 8 | 0.0000 | 0.0938 | 0.2500 |
