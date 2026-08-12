#!/usr/bin/env python
"""Train the Week-3 Lifetime head: BGE embeddings -> CoxPH survival model.

    python scripts/train.py --data-dir data/processed --emb-dir artifacts/embeddings \
        --out artifacts/survival_model_net.pt --seed 42
"""
import argparse
from pathlib import Path

import torchtuples as tt

from memorylife.data.datasets import load_split
from memorylife.encoders.cache import ensure_embeddings
from memorylife.heads.survival import build_survival_net
from memorylife.losses.cox_partial import build_cox_model
from memorylife.models.checkpoint import save_survival_model
from memorylife.utils.seeding import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--out", default="artifacts/survival_model_net.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden1", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=64)
    ap.add_argument("--dropout1", type=float, default=0.2)
    ap.add_argument("--dropout2", type=float, default=0.1)
    ap.add_argument("--emb-model-name", default=None,
                     help="override the sentence-transformers encoder, e.g. BAAI/bge-large-en-v1.5 (Week-4 encoder ablation)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seed_everything(args.seed)

    ensure_embeddings(args.data_dir, args.emb_dir, ["train", "val"], device=args.device,
                       model_name=args.emb_model_name)

    train = load_split(args.data_dir, args.emb_dir, "train")
    val = load_split(args.data_dir, args.emb_dir, "val")

    x_train = train["embeddings"].astype("float32")
    x_val = val["embeddings"].astype("float32")
    y_train = (train["durations"], train["events"])
    y_val = (val["durations"], val["events"])

    net = build_survival_net(x_train.shape[1], hidden1=args.hidden1, hidden2=args.hidden2,
                              dropout1=args.dropout1, dropout2=args.dropout2)
    model = build_cox_model(net, lr=args.lr, weight_decay=args.weight_decay)

    callbacks = [tt.callbacks.EarlyStopping(patience=args.patience)]
    model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=True,
        val_data=(x_val, y_val),
        val_batch_size=args.batch_size,
    )
    model.compute_baseline_hazards()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_survival_model(model, args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    main()
