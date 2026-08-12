#!/usr/bin/env python
"""Week-4: multi-seed reproduction of the Week-3 result.

Week 3 reported a single number (seed=42). That's not enough to claim "our
model beats the baselines" -- it could be seed luck. For each seed in
experiments/seeds.txt this retrains the survival model and re-fits
bucket_classifier (the only two methods with real training-time stochasticity;
heuristic_ttl and the LLM-prompted baselines are deterministic/non-trained and
are reused as-is from artifacts/scores/ to avoid re-spending API tokens for no
reason), re-evaluates on val/test, and saves each run under
experiments/main/survival_seed<K>/ (config + metrics + log, per the
experiments/README.md convention). Aggregates across seeds into mean +/- std.

    python scripts/run_seed_sweep.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIXED_METHODS = ("heuristic_ttl", "llm_prompted_ttl", "chatgpt_prompted_ttl", "gemini_prompted_ttl")


def run(cmd, log_path: Path) -> None:
    print(f"$ {' '.join(cmd)}")
    with open(log_path, "a") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log.write(proc.stdout)
        print(proc.stdout[-2000:])
        proc.check_returncode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-file", default="experiments/seeds.txt")
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--fixed-scores-dir", default="artifacts/scores",
                     help="where the seed-independent baseline scores already live")
    ap.add_argument("--out-dir", default="experiments/main")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()

    seeds = [int(s) for s in Path(args.seeds_file).read_text().split()]
    python = sys.executable
    per_seed_tables = []

    for seed in seeds:
        run_dir = Path(args.out_dir) / f"survival_seed{seed}"
        scores_dir = run_dir / "scores"
        results_dir = run_dir / "results"
        model_path = run_dir / "survival_model_net.pt"
        log_path = run_dir / "run.log"
        scores_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")

        run([python, "scripts/train.py", "--data-dir", args.data_dir, "--emb-dir", args.emb_dir,
             "--out", str(model_path), "--seed", str(seed), "--device", args.device], log_path)

        run([python, "scripts/run_baseline.py", "--method", "bucket_classifier",
             "--data-dir", args.data_dir, "--emb-dir", args.emb_dir,
             "--out-dir", str(scores_dir), "--splits", *args.splits,
             "--seed", str(seed), "--device", args.device], log_path)

        for method in FIXED_METHODS:
            for split in args.splits:
                src = Path(args.fixed_scores_dir) / f"{method}_{split}.json"
                if src.exists():
                    shutil.copy(src, scores_dir / src.name)

        run([python, "scripts/evaluate.py", "--data-dir", args.data_dir, "--emb-dir", args.emb_dir,
             "--model", str(model_path), "--scores-dir", str(scores_dir),
             "--out-dir", str(results_dir), "--splits", *args.splits, "--device", args.device], log_path)

        (run_dir / "config.json").write_text(json.dumps({"seed": seed, "splits": args.splits}, indent=2))

        df = pd.read_csv(results_dir / "raw" / "week3_results_raw.csv")
        df["seed"] = seed
        per_seed_tables.append(df)
        print(f"seed {seed}: done -> {run_dir}")

    all_df = pd.concat(per_seed_tables, ignore_index=True)
    all_df.to_csv(Path(args.out_dir) / "all_seeds_raw.csv", index=False)

    agg = (
        all_df.groupby(["split", "method"])["c_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "c_index_mean", "std": "c_index_std", "count": "n_seeds"})
        .sort_values(["split", "c_index_mean"], ascending=[True, False])
    )
    agg["c_index_std"] = agg["c_index_std"].fillna(0.0)

    out_tables = ROOT / "results" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_tables / "week4_multiseed_results.csv", index=False)

    md_lines = ["| Split | Method | C-index (mean +/- std over 5 seeds) | N seeds |", "|---|---|---|---|"]
    for _, r in agg.iterrows():
        md_lines.append(f"| {r['split']} | {r['method']} | {r['c_index_mean']:.4f} +/- {r['c_index_std']:.4f} | {int(r['n_seeds'])} |")
    (out_tables / "week4_multiseed_results.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("\n== multi-seed summary ==")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
