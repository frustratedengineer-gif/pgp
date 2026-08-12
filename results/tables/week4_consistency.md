| Method | Mean CV | Median CV | % records with CV > 0.5 | N records |
|---|---|---|---|---|
| Our survival model (CoxPH on BGE embeddings) | 0.0000 | 0.0000 | 0.0 | (deterministic given fixed weights, not re-run here) |
| LLM-prompted TTL (Gemini) | 0.3727 | 0.4351 | 23.0% | 100 |
| LLM-prompted TTL (ChatGPT / GPT-4o) | 0.3845 | 0.3206 | 44.0% | 100 |
| LLM-prompted TTL (local Qwen2.5-7B) | 0.7456 | 0.6813 | 85.0% | 100 |