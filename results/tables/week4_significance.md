| Split | Method | C-index 95% CI | Our model - method (95% CI) | p (one-sided, ours <= method) |
|---|---|---|---|---|
| val | Our survival model (CoxPH on BGE embeddings) | [0.6808, 0.7455] | - | - |
| val | LLM-prompted TTL (local Qwen2.5-7B) | [0.5389, 0.6014] | +0.1421 [+0.1052, +0.1808] | 0.0000 |
| val | Day/week/permanent classifier | [0.5884, 0.6492] | +0.0942 [+0.0538, +0.1336] | 0.0000 |
| val | Recency-frequency heuristic | [0.4508, 0.5216] | +0.2278 [+0.1853, +0.2723] | 0.0000 |
| val | LLM-prompted TTL (ChatGPT / GPT-4o) | [0.4950, 0.5631] | +0.1830 [+0.1390, +0.2285] | 0.0000 |
| val | LLM-prompted TTL (Gemini) | [0.3953, 0.4633] | +0.2845 [+0.2369, +0.3323] | 0.0000 |
| test | Our survival model (CoxPH on BGE embeddings) | [0.6894, 0.7515] | - | - |
| test | LLM-prompted TTL (local Qwen2.5-7B) | [0.4885, 0.5521] | +0.2013 [+0.1577, +0.2444] | 0.0000 |
| test | Day/week/permanent classifier | [0.5993, 0.6614] | +0.0916 [+0.0529, +0.1284] | 0.0000 |
| test | Recency-frequency heuristic | [0.4422, 0.5103] | +0.2462 [+0.2036, +0.2891] | 0.0000 |
| test | LLM-prompted TTL (ChatGPT / GPT-4o) | [0.5085, 0.5761] | +0.1801 [+0.1340, +0.2235] | 0.0000 |
| test | LLM-prompted TTL (Gemini) | [0.4450, 0.5162] | +0.2413 [+0.1940, +0.2845] | 0.0000 |