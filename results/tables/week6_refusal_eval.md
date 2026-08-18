# Refusal behavior on LoCoMo's adversarial (category-5) questions

Same methodology as REMem (ICLR 2026) Table 6: precision/recall/F1 of correctly refusing genuinely unanswerable questions, pooled across all 10 LoCoMo conversations at matched storage budget per policy. Ground truth unanswerable = category-5 adversarial QA pairs (no real `answer` field, only a plausible-sounding wrong `adversarial_answer` decoy).

N adversarial questions per policy: 120. N answerable (non-adversarial) questions per policy: 120.

| Policy | # Refusals | Precision | Recall | F1 |
|---|---|---|---|---|
| no_forget | 149 | 0.718 | 0.892 | 0.796 |
| fifo | 166 | 0.639 | 0.883 | 0.741 |
| lru | 151 | 0.709 | 0.892 | 0.790 |
| ours | 165 | 0.648 | 0.892 | 0.751 |
| ours_utility | 146 | 0.726 | 0.883 | 0.797 |

Token usage (includes cache hits at face value): {'prompt_tokens': 539499, 'completion_tokens': 8314, 'total_tokens': 547813, 'calls': 1200}
