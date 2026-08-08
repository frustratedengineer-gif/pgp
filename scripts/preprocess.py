#!/usr/bin/env python
"""Derive (T, delta) survival targets for each split.

    python scripts/preprocess.py --raw-dir data/raw --out-dir data/processed
"""
import argparse
from pathlib import Path

from memorylife.data.event_labeling import label_records
from memorylife.data.splits import assert_no_leakage
from memorylife.utils.io import load_jsonl, write_jsonl

SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--raw-filenames", nargs=3, default=["train.jsonl", "val.jsonl", "test.jsonl"],
                     help="filenames within --raw-dir for train/val/test, in that order")
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_records = {}
    for split, fname in zip(SPLITS, args.raw_filenames):
        records = load_jsonl(raw_dir / fname)
        split_records[split] = records

    assert_no_leakage(split_records)

    for split, records in split_records.items():
        label_records(records)
        write_jsonl(records, out_dir / f"{split}_survival.jsonl")
        n = len(records)
        n_events = sum(r["event_observed"] for r in records)
        print(f"{split}: n={n} event_rate={n_events/n:.1%} -> {out_dir / f'{split}_survival.jsonl'}")


if __name__ == "__main__":
    main()
