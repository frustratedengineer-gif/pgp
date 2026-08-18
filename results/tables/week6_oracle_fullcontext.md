# Oracle and Full-Context reference points (reviewer gap)

Same methodology as REMem (ICLR 2026): `oracle` = answer given ONLY the QA pair's own gold-evidence memories (no eviction, no retrieval noise -- the true ceiling); `full_context` = answer given the ENTIRE conversation's memory store, uncapped by the usual top-5 retrieval (isolates retrieval-k cost from eviction cost). Compare against `no_forget` in week6_downstream_qa_q0.2_ranked_pilot.md, which retains everything but still goes through top-5 retrieval.

| Benchmark | Policy | N | Mean EM | Mean F1 | Mean BLEU-1 |
|---|---|---|---|---|---|
| locomo | full_context | 30 | 0.1333 | 0.2980 | 0.2646 |
| locomo | oracle | 107 | 0.1589 | 0.3473 | 0.2873 |
| longmemeval | full_context | 25 | 0.0800 | 0.1525 | 0.1052 |
| longmemeval | oracle | 22 | 0.1364 | 0.1814 | 0.1407 |

Token usage (includes cache hits at face value): {'prompt_tokens': 399267, 'completion_tokens': 1087, 'total_tokens': 400354, 'calls': 184}
