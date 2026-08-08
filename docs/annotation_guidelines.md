# Annotation Guidelines

Not applicable yet. All labels (`duration_days`, `event_observed`,
`censor_reason`) are derived programmatically from timestamps already
present in the source records (`src/memorylife/data/event_labeling.py`,
`censoring.py`) -- no human annotation step exists in the pipeline so far.

If a human validation pass is added later (e.g. spot-checking that
`lifecycle_event` labels on the real LoCoMo/LongMemEval subset are
correct), document the guidelines and inter-annotator agreement here.
