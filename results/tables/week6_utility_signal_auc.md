# Does utility_prob (or predicted_ttl_days) actually predict QA-evidence relevance?

Free, no-LLM-calls diagnostic: AUC of each signal predicting `is_evidence` (was this memory ever cited as evidence for a LoCoMo QA pair in its own conversation), pooled across all LoCoMo memories and all 10 conversations. 0.5 = random, 1.0 = perfect ranking -- same scale as the Week-5 Future-Utility head's own validation AUC (0.71-0.77) and the Week-3/4 survival model's C-index.

Pooled: n=2536 memories, 1171 (46.2%) are evidence for at least one QA pair.

| Signal | Pooled AUC (predicts is_evidence) |
|---|---|
| utility_prob (Future-Utility head) | 0.6709 |
| predicted_ttl_days (Lifetime head, median quantile) | 0.2852 |

## Per-conversation (conversations with both evidence and non-evidence memories only)

| Conversation | N memories | N evidence | AUC(utility_prob) | AUC(predicted_ttl_days) |
|---|---|---|---|---|
| conv-26 | 184 | 103 | 0.6997 | 0.2312 |
| conv-30 | 169 | 57 | 0.6386 | 0.2011 |
| conv-41 | 324 | 119 | 0.7005 | 0.3309 |
| conv-42 | 266 | 140 | 0.6142 | 0.2039 |
| conv-43 | 267 | 125 | 0.6263 | 0.2213 |
| conv-44 | 276 | 110 | 0.6496 | 0.4115 |
| conv-47 | 268 | 103 | 0.6398 | 0.1730 |
| conv-48 | 289 | 149 | 0.6342 | 0.2863 |
| conv-49 | 239 | 145 | 0.7577 | 0.2414 |
| conv-50 | 254 | 120 | 0.6301 | 0.1954 |