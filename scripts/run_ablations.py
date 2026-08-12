#!/usr/bin/env python
"""Week-4: lightweight ablations. Feature extractors, fusion, and the other
3 heads (importance/utility/action) aren't built yet -- that's Week 5's
"Full System" scope, not Week 4's. What's available now, cheaply, on top of
the Week-3 survival head:

  1. Encoder choice: BGE-base (768d, Week-3 default) vs BGE-large (1024d).
  2. Survival-head hyperparameter sensitivity: hidden width, dropout, lr.

Each variant is averaged over multiple seeds (not a single run), matching
the multi-seed convention from scripts/run_seed_sweep.py.

    python scripts/run_ablations.py
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def run(cmd) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def train_and_eval(python, seed, out_dir, splits, device, data_dir, emb_dir,
                    extra_train_args=None, eval_extra_args=None) -> dict:
    extra_train_args = extra_train_args or []
    eval_extra_args = eval_extra_args or []
    model_path = out_dir / f"model_seed{seed}.pt"
    run([python, "scripts/train.py", "--data-dir", data_dir, "--emb-dir", emb_dir,
         "--out", str(model_path), "--seed", str(seed), "--device", device, *extra_train_args])
    metrics_path = out_dir / f"metrics_seed{seed}.json"
    run([python, "scripts/eval_ours_only.py", "--data-dir", data_dir, "--emb-dir", emb_dir,
         "--model", str(model_path), "--splits", *splits, "--device", device,
         "--out", str(metrics_path), *eval_extra_args])
    return json.loads(metrics_path.read_text())


def aggregate(rows, group_col) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = df.groupby([group_col, "split"])["c_index"].agg(["mean", "std", "count"]).reset_index()
    agg["std"] = agg["std"].fillna(0.0)
    return agg


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")


def write_table(agg: pd.DataFrame, group_col: str, out_tables: Path, name: str) -> None:
    agg.to_csv(out_tables / f"{name}.csv", index=False)
    md = [f"| {group_col.capitalize()} | Split | C-index (mean +/- std) | N seeds |", "|---|---|---|---|"]
    for _, r in agg.sort_values(["split", "mean"], ascending=[True, False]).iterrows():
        md.append(f"| {r[group_col]} | {r['split']} | {r['mean']:.4f} +/- {r['std']:.4f} | {int(r['count'])} |")
    (out_tables / f"{name}.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--out-root", default="experiments/ablation")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encoder-seeds", nargs="+", type=int, default=[13, 42, 1337, 2024, 7])
    ap.add_argument("--hparam-seeds", nargs="+", type=int, default=[13, 42, 1337])
    args = ap.parse_args()
    python = sys.executable
    out_tables = ROOT / "results" / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    # -------- 1. encoder ablation: BGE-base vs BGE-large --------
    encoder_rows = []
    for name, model_name, emb_dir in [
        ("bge-base-en-v1.5 (768d, Week-3 default)", None, args.emb_dir),
        ("bge-large-en-v1.5 (1024d)", "BAAI/bge-large-en-v1.5", "artifacts/embeddings_bge_large"),
    ]:
        out_dir = Path(args.out_root) / "encoder" / slug(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_name_args = ["--emb-model-name", model_name] if model_name else []
        for seed in args.encoder_seeds:
            metrics = train_and_eval(python, seed, out_dir, args.splits, args.device,
                                      args.data_dir, emb_dir, extra_train_args=model_name_args,
                                      eval_extra_args=model_name_args)
            for split, c in metrics.items():
                encoder_rows.append({"encoder": name, "split": split, "c_index": c, "seed": seed})
    encoder_agg = aggregate(encoder_rows, "encoder")
    write_table(encoder_agg, "encoder", out_tables, "week4_ablation_encoder")

    # -------- 2. survival-head hyperparameter sensitivity --------
    base = {"hidden1": 256, "hidden2": 64, "dropout1": 0.2, "dropout2": 0.1, "lr": 1e-3}
    variants = {
        "base (Week-3 default)": {},
        "hidden1=128": {"hidden1": 128},
        "hidden1=512": {"hidden1": 512},
        "dropout1=0.0": {"dropout1": 0.0},
        "dropout1=0.4": {"dropout1": 0.4},
        "lr=3e-4": {"lr": 3e-4},
        "lr=3e-3": {"lr": 3e-3},
    }
    hparam_rows = []
    for name, overrides in variants.items():
        cfg = {**base, **overrides}
        out_dir = Path(args.out_root) / "sensitivity" / slug(name)
        out_dir.mkdir(parents=True, exist_ok=True)
        train_extra = [
            "--hidden1", str(cfg["hidden1"]), "--hidden2", str(cfg["hidden2"]),
            "--dropout1", str(cfg["dropout1"]), "--dropout2", str(cfg["dropout2"]),
            "--lr", str(cfg["lr"]),
        ]
        eval_extra = ["--hidden1", str(cfg["hidden1"]), "--hidden2", str(cfg["hidden2"]),
                      "--dropout1", str(cfg["dropout1"]), "--dropout2", str(cfg["dropout2"])]
        for seed in args.hparam_seeds:
            metrics = train_and_eval(python, seed, out_dir, args.splits, args.device,
                                      args.data_dir, args.emb_dir, extra_train_args=train_extra,
                                      eval_extra_args=eval_extra)
            for split, c in metrics.items():
                hparam_rows.append({"variant": name, "split": split, "c_index": c, "seed": seed})
    hparam_agg = aggregate(hparam_rows, "variant")
    write_table(hparam_agg, "variant", out_tables, "week4_ablation_hparams")

    print("\n== encoder ablation ==")
    print(encoder_agg.to_string(index=False))
    print("\n== hyperparameter sensitivity ==")
    print(hparam_agg.to_string(index=False))


if __name__ == "__main__":
    main()
