# Automated label-consistency audit (full coverage, not a sample)

Recomputes `duration_days` from raw timestamps under the Section 3.3 censoring rules
for every record and flags mismatches beyond a 0.01-day (~15 min) floating-point
tolerance. Catches arithmetic/derivation bugs; does NOT validate the semantic
correctness of event_observed/censor_reason itself -- see the separate human-judgment
spot-check (scripts/sample_for_human_validation.py) for that.

**Process note, disclosed rather than silently fixed**: the first version of this audit used the wrong reference timestamp for LongMemEval's case-3 (censored, no probes) rule -- it used the max `injected_at` among a conversation's *extracted* memories, which undercounts the true conversation-end time whenever later turns didn't yield an extracted memory. That first pass showed 2,601/10,152 (25.6%) mismatches. Cross-checking one flagged record against `data/raw/longmemeval_s_cleaned.json` showed `injected_at + stored duration_days` reproduces that conversation's own `question_date` to the minute (2023-03-10T06:59 + 22.6132d = 2023-04-01T21:42:00, exactly matching `gpt4_74aed68e`'s question_date) -- i.e. the ORIGINAL labels were right; this audit script's assumption was wrong. Fixed to use `question_date` for LongMemEval case-3 records (see `load_lme_question_dates()`), which is the number below.

**Total records audited: 10152. Mismatches after the fix: 470 (4.63%), all in LongMemEval, none in LoCoMo or synthetic.**

| Case | N records |
|---|---|
| 1_event_observed | 3755 |
| 2_censored_with_probes | 2066 |
| 3_censored_no_probes | 4331 |

Of the 470 residual mismatches: 449 are small (<=1 day) or match a `duration_days=0.01` floor-clamp pattern consistent with a defensive minimum-duration guard (plausible, not confirmed by reading the original extraction code, which is not in this repo -- see `data/README.md`'s known gap). The remaining **21 records** (0.21% of the full dataset) have a genuine, unexplained gap >1 day (up to 53.2 days) between the stored label and what this audit recomputes from raw timestamps. This was not resolved further -- doing so would mean reverse-engineering pipeline logic not present in this repo, out of scope for this check -- and is reported here as a genuine, disclosed residual rather than hidden or rounded away.

## Unexplained residual mismatches (>1 day, not floor-clamp related)

| memory_id | split | case | stored | recomputed | diff (days) |
|---|---|---|---|---|---|
| real_lme_0373_026 | val | 3_censored_no_probes | 61.2535 | 62.7590 | -1.5055 |
| real_lme_0283_007 | val | 3_censored_no_probes | 3.9806 | 25.8813 | -21.9007 |
| real_lme_0373_011 | val | 3_censored_no_probes | 171.2972 | 172.8028 | -1.5056 |
| real_lme_0227_032 | val | 3_censored_no_probes | 2.0826 | 4.2264 | -2.1438 |
| real_lme_0373_010 | val | 3_censored_no_probes | 177.9403 | 179.4458 | -1.5055 |
| real_lme_0267_027 | val | 3_censored_no_probes | 6.7583 | 59.9715 | -53.2132 |
| real_lme_0227_039 | val | 3_censored_no_probes | 0.9479 | 3.0917 | -2.1438 |
| real_lme_0103_032 | val | 3_censored_no_probes | 1.4410 | 4.1160 | -2.6750 |
| real_lme_0103_030 | val | 3_censored_no_probes | 1.6236 | 4.2986 | -2.6750 |
| real_lme_0103_028 | val | 3_censored_no_probes | 2.3507 | 5.0257 | -2.6750 |
| real_lme_0227_016 | val | 3_censored_no_probes | 5.0403 | 7.1840 | -2.1437 |
| real_lme_0267_024 | val | 3_censored_no_probes | 9.8979 | 63.1111 | -53.2132 |
| real_lme_0283_020 | val | 3_censored_no_probes | 0.6785 | 22.5792 | -21.9007 |
| real_lme_0227_028 | val | 3_censored_no_probes | 3.1896 | 5.3333 | -2.1437 |
| real_lme_0373_005 | val | 3_censored_no_probes | 249.4674 | 250.9729 | -1.5055 |
| real_lme_0283_017 | val | 3_censored_no_probes | 0.5451 | 22.4458 | -21.9007 |
| real_lme_0283_023 | val | 3_censored_no_probes | 0.9368 | 22.8375 | -21.9007 |
| real_lme_0373_036 | val | 3_censored_no_probes | 11.8465 | 13.3521 | -1.5056 |
| real_lme_0283_008 | val | 3_censored_no_probes | 4.2125 | 26.1132 | -21.9007 |
| real_lme_0227_034 | val | 3_censored_no_probes | 1.9438 | 4.0875 | -2.1437 |
| real_lme_0267_030 | val | 3_censored_no_probes | 3.9583 | 57.1715 | -53.2132 |