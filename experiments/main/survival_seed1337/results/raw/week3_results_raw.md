| Split | Method | C-index | N |
|---|---|---|---|
| test | Our survival model (CoxPH on BGE embeddings) | 0.7455 | 916 |
| test | Day/week/permanent classifier | 0.6298 | 916 |
| test | LLM-prompted TTL (ChatGPT / GPT-4o) | 0.5411 | 916 |
| test | LLM-prompted TTL (local Qwen2.5-7B) | 0.5207 | 916 |
| test | LLM-prompted TTL (Gemini) | 0.4806 | 916 |
| test | Recency-frequency heuristic | 0.4753 | 916 |
| val | Our survival model (CoxPH on BGE embeddings) | 0.7304 | 939 |
| val | Day/week/permanent classifier | 0.6195 | 939 |
| val | LLM-prompted TTL (local Qwen2.5-7B) | 0.5713 | 939 |
| val | LLM-prompted TTL (ChatGPT / GPT-4o) | 0.5312 | 939 |
| val | Recency-frequency heuristic | 0.4849 | 939 |
| val | LLM-prompted TTL (Gemini) | 0.4284 | 939 |