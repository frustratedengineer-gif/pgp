# Does LoCoMo's QA evidence skew recent? (closes a Limitations gap)

Tests the Section 7 working hypothesis directly: if evidence-linked memories are
systematically younger (smaller age relative to the conversation's own last timestamp)
than non-evidence memories, that would mechanically favor recency-ranked policies
(`fifo`/`lru`) over any policy that ignores recency -- independent of anything
`ours`/`ours_utility` do differently.

Pooled across all 10 LoCoMo conversations, 2536 total memories (1171 evidence, 1365 non-evidence).

| Group | N | Mean age (days) | Median age (days) |
|---|---|---|---|
| Evidence-linked | 1171 | 104.2 | 94.5 |
| Non-evidence | 1365 | 97.3 | 86.0 |

Unpaired bootstrap (10,000 resamples, independent groups), difference reported as
(non-evidence age) - (evidence age) -- positive means evidence memories ARE younger,
confirming the hypothesis; a CI straddling 0 means not confirmed at this sample size:

mean diff = -6.87 days, 95% CI [-12.78, -1.02], p (non-evidence <= evidence, i.e. hypothesis holds) = 0.0100

**Verdict: REFUTED (significant, reversed direction)** at the 95% level.
The hypothesis is not just unconfirmed -- it is significantly wrong in this dataset. Evidence-linked memories are on average *older* than non-evidence memories (104.2 vs. 97.3 days, a statistically significant ~6.9-day gap), not younger. This rules out recency-skew as the explanation for the residual `ours` vs. `lru`/`fifo` gap in Section 6.4 -- if anything, a naive recency policy is working *against* a mild anti-recency skew in LoCoMo's evidence and still winning, which makes the structural argument for ranked top-N selection (Section 6.5's actual explanation) stronger, not weaker: the advantage comes from the ranking *mechanism*, not from recency happening to align with what's needed.