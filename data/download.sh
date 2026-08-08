#!/usr/bin/env bash
# TODO (known gap, see data/README.md): this should download/build LoCoMo,
# LongMemEval, and the synthetic conversations from scratch and produce
# data/raw/{train,val,test}.jsonl. That generation code
# (src/memorylife/data/build_benchmark.py) has not been written yet -- the
# Week 1-3 results were produced from raw files that arrived already built.
#
# Until build_benchmark.py exists, this script just verifies checksums for
# whatever is already sitting in data/raw/ (e.g. copied back in manually)
# rather than pretending to fetch anything.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f data/raw/train.jsonl ]; then
  echo "data/raw/train.jsonl not found." >&2
  echo "This script cannot fetch it yet -- see the TODO at the top of data/download.sh" >&2
  exit 1
fi

echo "Verifying checksums against data/checksums.sha256 ..."
sha256sum -c data/checksums.sha256
