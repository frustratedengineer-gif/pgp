| Split | Method | C-index (mean +/- std over 5 seeds) | N seeds |
|---|---|---|---|
| test | Our survival model (CoxPH on BGE embeddings) | 0.7312 +/- 0.0131 | 5 |
| test | Day/week/permanent classifier | 0.6298 +/- 0.0000 | 5 |
| test | LLM-prompted TTL (ChatGPT / GPT-4o) | 0.5411 +/- 0.0000 | 5 |
| test | LLM-prompted TTL (local Qwen2.5-7B) | 0.5207 +/- 0.0000 | 5 |
| test | LLM-prompted TTL (Gemini) | 0.4806 +/- 0.0000 | 5 |
| test | Recency-frequency heuristic | 0.4753 +/- 0.0000 | 5 |
| val | Our survival model (CoxPH on BGE embeddings) | 0.7237 +/- 0.0063 | 5 |
| val | Day/week/permanent classifier | 0.6195 +/- 0.0000 | 5 |
| val | LLM-prompted TTL (local Qwen2.5-7B) | 0.5713 +/- 0.0000 | 5 |
| val | LLM-prompted TTL (ChatGPT / GPT-4o) | 0.5312 +/- 0.0000 | 5 |
| val | Recency-frequency heuristic | 0.4849 +/- 0.0000 | 5 |
| val | LLM-prompted TTL (Gemini) | 0.4284 +/- 0.0000 | 5 |