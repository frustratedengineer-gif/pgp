#!/usr/bin/env python
"""Week-5: train the joint multi-task model (Lifetime + Action +
Future-utility heads sharing one fused representation z) and check whether
adding features/fusion/multi-task supervision improves on the Week-3/4 lone
survival head's C-index (results/tables/week4_multiseed_results.md:
0.7312 +/- 0.0131 test, single-head, embedding-only).

Not reusing pycox's CoxPH.fit() wrapper (scripts/train.py's approach) --
that wrapper drives its own training loop and can't share gradients with
the action/utility heads through one fusion backbone. This is a custom
loop using pycox's standalone cox_ph_loss (see models/multitask.py).

    python scripts/train_joint.py --fusion gated --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from memorylife.data.datasets import load_split
from memorylife.encoders.cache import ensure_embeddings
from memorylife.evaluation.survival_metrics import c_index_for_split
from memorylife.features.pipeline import ensure_features, load_features, slices_path
from memorylife.heads.action import ACTION_LABELS, action_label_from_lifecycle_event
from memorylife.heads.future_utility import has_utility_label, utility_label_from_lifecycle_event
from memorylife.losses.action_loss import build_action_loss
from memorylife.losses.utility_loss import build_utility_loss
from memorylife.models.checkpoint import save_joint_model
from memorylife.models.joint_predictor import JointLifecyclePredictor
from memorylife.models.multitask import LossWeights, compute_joint_loss
from memorylife.utils.seeding import seed_everything


def build_supervision(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_labels = np.array([action_label_from_lifecycle_event(r["lifecycle_event"]) for r in records],
                              dtype=np.int64)
    utility_mask = np.array([has_utility_label(r["lifecycle_event"]) for r in records], dtype=bool)
    utility_labels = np.zeros(len(records), dtype=np.float32)
    for i, r in enumerate(records):
        if utility_mask[i]:
            utility_labels[i] = utility_label_from_lifecycle_event(r["lifecycle_event"])
    return action_labels, utility_labels, utility_mask


def load_split_tensors(data_dir, emb_dir, feat_dir, split, device):
    split_data = load_split(data_dir, emb_dir, split)
    feats = load_features(feat_dir, split)
    action_labels, utility_labels, utility_mask = build_supervision(split_data["records"])
    return {
        "ids": split_data["ids"],
        "records": split_data["records"],
        "embedding": torch.tensor(split_data["embeddings"], dtype=torch.float32, device=device),
        "features": torch.tensor(feats, dtype=torch.float32, device=device),
        "durations": torch.tensor(split_data["durations"], dtype=torch.float32, device=device),
        "events": torch.tensor(split_data["events"], dtype=torch.float32, device=device),
        "action_labels": torch.tensor(action_labels, dtype=torch.long, device=device),
        "utility_labels": torch.tensor(utility_labels, dtype=torch.float32, device=device),
        "utility_mask": torch.tensor(utility_mask, dtype=torch.bool, device=device),
    }


def evaluate_split(model, batch, split_name) -> dict:
    model.eval()
    with torch.no_grad():
        out = model(batch["embedding"], batch["features"])
    scores = {mid: -float(s) for mid, s in zip(batch["ids"], out["log_hazard"].cpu().numpy())}
    split_for_cindex = {"ids": batch["ids"], "durations": batch["durations"].cpu().numpy(),
                         "events": batch["events"].cpu().numpy()}
    c = c_index_for_split(scores, split_for_cindex)

    action_pred = out["action_logits"].argmax(dim=-1).cpu().numpy()
    action_true = batch["action_labels"].cpu().numpy()
    action_acc = float((action_pred == action_true).mean())

    mask = batch["utility_mask"].cpu().numpy()
    utility_acc = None
    if mask.any():
        utility_pred = (torch.sigmoid(out["utility_logit"]).cpu().numpy() > 0.5)
        utility_true = batch["utility_labels"].cpu().numpy().astype(bool)
        utility_acc = float((utility_pred[mask] == utility_true[mask]).mean())

    utility_acc_str = f"{utility_acc:.4f}" if utility_acc is not None else "n/a"
    print(f"  {split_name}: c_index={c:.4f}  action_acc={action_acc:.4f}  utility_acc={utility_acc_str}")
    return {"c_index": c, "action_acc": action_acc, "utility_acc": utility_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--feat-dir", default="artifacts/features")
    ap.add_argument("--out", default="artifacts/joint_model.pt")
    ap.add_argument("--fusion", default="gated", choices=["concat", "gated"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--survival-weight", type=float, default=1.0)
    ap.add_argument("--action-weight", type=float, default=0.5)
    ap.add_argument("--utility-weight", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    ensure_embeddings(args.data_dir, args.emb_dir, ["train", "val", "test"], device=device)
    ensure_features(args.data_dir, args.emb_dir, args.feat_dir, ["train", "val", "test"], device=device)

    train = load_split_tensors(args.data_dir, args.emb_dir, args.feat_dir, "train", device)
    val = load_split_tensors(args.data_dir, args.emb_dir, args.feat_dir, "val", device)
    test = load_split_tensors(args.data_dir, args.emb_dir, args.feat_dir, "test", device)

    embedding_dim = train["embedding"].shape[1]
    feature_dim = train["features"].shape[1]
    model = JointLifecyclePredictor(embedding_dim, feature_dim, fusion_name=args.fusion).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    action_loss_fn = build_action_loss(train["action_labels"].cpu().numpy(), len(ACTION_LABELS)).to(device)
    train_utility_mask_np = train["utility_mask"].cpu().numpy()
    utility_loss_fn = build_utility_loss(
        train["utility_labels"].cpu().numpy()[train_utility_mask_np]
    ).to(device)
    weights = LossWeights(survival=args.survival_weight, action=args.action_weight, utility=args.utility_weight)

    best_val_loss = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train["embedding"], train["features"])
        loss, breakdown = compute_joint_loss(
            out, train["durations"], train["events"], train["action_labels"],
            train["utility_labels"], train["utility_mask"], action_loss_fn, utility_loss_fn, weights,
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(val["embedding"], val["features"])
            val_loss, val_breakdown = compute_joint_loss(
                val_out, val["durations"], val["events"], val["action_labels"],
                val["utility_labels"], val["utility_mask"], action_loss_fn, utility_loss_fn, weights,
            )

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: train_loss={breakdown['total_loss']:.4f} "
                  f"(surv={breakdown['survival_loss']:.4f} act={breakdown['action_loss']:.4f} "
                  f"util={breakdown['utility_loss']:.4f})  val_loss={val_breakdown['total_loss']:.4f}")

        if val_breakdown["total_loss"] < best_val_loss - 1e-4:
            best_val_loss = val_breakdown["total_loss"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\n== final evaluation ==")
    evaluate_split(model, val, "val")
    evaluate_split(model, test, "test")

    config = {
        "embedding_dim": embedding_dim, "feature_dim": feature_dim, "fusion_name": args.fusion,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_joint_model(model, args.out, config)
    print(f"saved joint model -> {args.out}")


if __name__ == "__main__":
    main()
