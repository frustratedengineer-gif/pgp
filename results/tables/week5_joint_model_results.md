| Fusion | Split | C-index (survival) | Action accuracy | Utility accuracy | N seeds |
|---|---|---|---|---|---|
| concat | test | 0.7553 +/- 0.0045 | 0.8642 +/- 0.0005 | 0.6170 +/- 0.0072 | 3 |
| gated | test | 0.7304 +/- 0.0082 | 0.8384 +/- 0.0070 | 0.6165 +/- 0.0044 | 3 |
| concat | val | 0.7468 +/- 0.0008 | 0.8690 +/- 0.0000 | 0.6847 +/- 0.0062 | 3 |
| gated | val | 0.7467 +/- 0.0067 | 0.8463 +/- 0.0051 | 0.6037 +/- 0.0018 | 3 |

Seeds: 13, 42, 1337 (subset of the Week-4 seed list -- 3 rather than 5, since
this is a single design-comparison check, not the headline result).

Reference point: the Week-3/4 lone survival head (embedding only, no
features/fusion/multi-task) got 0.7312 +/- 0.0131 test C-index over 5 seeds
(`results/tables/week4_multiseed_results.md`). Concat fusion + auxiliary
features + joint action/utility supervision reaches 0.7553 +/- 0.0045 test
C-index -- a real, consistent improvement (non-overlapping error bars),
achieved with a TIGHTER spread across seeds than the lone head had.

Concat beats gated fusion here, consistently, despite gated being the
"ours"/more-sophisticated mechanism per the architecture figure's colour
legend -- worth reporting as-is rather than assuming more parameters means
better. A plausible explanation: the learned gate has more capacity to
overfit on a dataset this size (~8.3K train records), and/or the fixed
Week-3/4 head sizes weren't re-tuned for the gated variant's different
effective input distribution. Not chased further given the Week-5 time
budget; noted as a natural follow-up.

Utility accuracy (~0.60-0.68) is modest relative to survival C-index and
action accuracy -- expected, since it's a genuinely hard binary prediction
(will this specific memory be referenced again?) trained on real but
comparatively coarse labels (see heads/future_utility.py's docstring on
what counts as a "usage" label).
