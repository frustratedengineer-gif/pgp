#!/usr/bin/env bash
# ~3 minute pipeline sanity check on data/samples/ (60 records, CPU-friendly).
# Skips the llm_prompted_ttl baseline (needs a running ollama server) --
# smoke-tests everything that doesn't depend on an external service.
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE_DIR=$(mktemp -d)
trap 'rm -rf "$SMOKE_DIR"' EXIT

python - "$SMOKE_DIR" <<'PY'
import json, sys
from pathlib import Path

smoke_dir = Path(sys.argv[1])
records = [json.loads(l) for l in open("data/samples/memories_sample.jsonl", encoding="utf-8")]
# reuse the same small pool for all 3 splits -- this is a pipeline smoke
# test, not a real train/val/test evaluation.
for split in ("train", "val", "test"):
    with open(smoke_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
PY

echo "== Preprocess (smoke) =="
python scripts/preprocess.py --raw-dir "$SMOKE_DIR" --out-dir "$SMOKE_DIR/processed"

echo "== Train (smoke: 5 epochs, CPU, small batch) =="
python scripts/train.py \
  --data-dir "$SMOKE_DIR/processed" --emb-dir "$SMOKE_DIR/emb" \
  --out "$SMOKE_DIR/model.pt" --device cpu --epochs 5 --patience 5 --batch-size 16

echo "== Baselines (smoke; llm_prompted_ttl skipped, needs ollama) =="
python scripts/run_baseline.py --method heuristic_ttl \
  --data-dir "$SMOKE_DIR/processed" --out-dir "$SMOKE_DIR/scores" --splits train val test
python scripts/run_baseline.py --method bucket_classifier \
  --data-dir "$SMOKE_DIR/processed" --emb-dir "$SMOKE_DIR/emb" --out-dir "$SMOKE_DIR/scores" --splits train val test

echo "Smoke test passed: preprocess -> train -> baselines ran end to end."
