# Downstream QA: EM + F1 + BLEU-1 + LLM-judge, together (reviewer gap)

Same q0.2 ranked-pilot predictions as week6_downstream_qa_q0.2_ranked_pilot.md / week6_judge_scores_*.md, with BLEU-1 (`memorylife.evaluation.qa_metrics.bleu1`, unigram precision x brevity penalty) computed fresh over the same already-collected predictions -- free, no new LLM calls. Matches REMem's own table shape (F1 + BLEU-1 + LLM-judge together).

| Benchmark | Policy | N | Mean EM | Mean F1 | Mean BLEU-1 | Mean Judge |
|---|---|---|---|---|---|---|
| locomo | fifo | 120 | 0.0417 | 0.1294 | 0.0915 | 0.1833 |
| locomo | lru | 120 | 0.0833 | 0.1998 | 0.1586 | 0.3417 |
| locomo | no_forget | 120 | 0.0917 | 0.1957 | 0.1562 | 0.3333 |
| locomo | ours | 120 | 0.0500 | 0.1575 | 0.1116 | 0.2000 |
| locomo | ours_utility | 120 | 0.0917 | 0.1949 | 0.1549 | 0.3500 |
| longmemeval | fifo | 25 | 0.0800 | 0.1558 | 0.1052 | 0.4000 |
| longmemeval | lru | 25 | 0.0800 | 0.1289 | 0.0878 | 0.2800 |
| longmemeval | no_forget | 25 | 0.0800 | 0.1525 | 0.1052 | 0.3600 |
| longmemeval | ours | 25 | 0.0800 | 0.1563 | 0.1066 | 0.3600 |
| longmemeval | ours_utility | 25 | 0.0800 | 0.1563 | 0.1066 | 0.3600 |