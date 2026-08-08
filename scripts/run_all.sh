#!/usr/bin/env bash
# Full reproduction of the Week 1-3 results (the only weeks that exist yet
# -- this does NOT run Week 4/5's calibration/consistency/cost or
# end-to-end QA experiments, since that code isn't written).
#
# Expected runtime on a single modern GPU: ~15 minutes, dominated by the
# LLM baseline over the full train split (~12 min). See
# docs/reproducibility.md for the exact per-step timings measured on
# 8x H200 (shared node).
#
# Prerequisite: data/raw/{train,val,test}.jsonl must already exist (see
# data/README.md -- this repo does not yet generate them from scratch).
# Prerequisite: an ollama server running locally with
# `ollama pull qwen2.5:7b-instruct` already done (see baselines/llm_prompted_ttl.py).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Preprocess: derive (T, delta) survival targets =="
python scripts/preprocess.py

echo "== Train: BGE embeddings + CoxPH survival model =="
python scripts/train.py

echo "== Baselines =="
python scripts/run_baseline.py --method heuristic_ttl
python scripts/run_baseline.py --method bucket_classifier
python scripts/run_baseline.py --method llm_prompted_ttl

echo "== Evaluate: C-index table =="
python scripts/evaluate.py

echo "Done. See results/tables/week3_results_table.md"
