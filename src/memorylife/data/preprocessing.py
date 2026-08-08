"""
Text cleaning / segmentation / speaker normalisation, ahead of
event_labeling.py.

Not yet needed: the Week 1-2 dataset (LoCoMo + LongMemEval + synthetic,
already reduced to one fact-statement per record) arrived pre-extracted, so
there has been no raw-dialogue cleaning step to write yet. See
docs/benchmark_card.md for what W1/W2 produced vs. what's still a gap
(the dialogue -> candidate-memory extraction code itself is not in this
repo -- see the README "known gaps" section).

JSONL I/O helpers live in utils/io.py, used by this module once there is
something to preprocess.
"""
