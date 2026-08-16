#!/usr/bin/env python
"""
Week-6 reviewer gap: `ours_utility` (rank eviction by the Future-Utility
head's P(retrieved again)) empirically beats the original TTL-threshold
policy and is statistically indistinguishable from the no_forget ceiling
(week6_downstream_significance.md) -- but nothing so far explains WHY.
This measures it directly: does utility_prob actually correlate with
"is this memory ever cited as QA evidence" on LoCoMo, independent of any
eviction policy or downstream LLM call?

Free (no LLM calls, no new training) -- reuses the already-trained
survival/joint model checkpoints purely for inference, and the same
evidence-linkage machinery `diagnose_eviction_evidence.py` already uses.

Label: for every memory in a LoCoMo conversation, is_evidence=1 if its
evidence_dia_id(s) appear in the `evidence` list of at least one
answerable (non-adversarial) QA pair for that conversation, else 0.
Computed over ALL memories (not just the ones that survive some
eviction policy), so this is a property of the SIGNAL, not of any
particular forgetting decision.

Reports pooled AUC of two candidate ranking signals against that label:
  - utility_prob (Future-Utility head) -- what `ours_utility` ranks by
  - predicted_ttl_days at the median quantile (Lifetime head's natural
    point estimate, independent of any eviction-policy quantile choice)
    -- what the original `ours` effectively ranks by

    python scripts/analyze_utility_signal.py --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from memorylife.data.build_benchmark import _dia_ids_of, load_all_processed, load_locomo_qa
from memorylife.inference.pipeline import build_memory_objects
from memorylife.models.checkpoint import load_joint_model, load_survival_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--feat-dir", default="artifacts/features")
    ap.add_argument("--locomo-path", default="data/raw/locomo10.json")
    ap.add_argument("--survival-model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--joint-model", default="artifacts/joint_model.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    processed = load_all_processed(args.data_dir, args.emb_dir, args.feat_dir)
    feature_slices = json.loads((Path(args.feat_dir) / "feature_slices.json").read_text())

    survival_model = load_survival_model(args.survival_model, 768)
    train_records = [v for v in processed["by_memory_id"].values() if v["split"] == "train"]
    train_emb = np.stack([v["embedding"] for v in train_records]).astype("float32")
    train_dur = np.array([v["record"]["duration_days"] for v in train_records], dtype=np.float32)
    train_ev = np.array([v["record"]["event_observed"] for v in train_records], dtype=np.float32)
    survival_model.compute_baseline_hazards(input=train_emb, target=(train_dur, train_ev))

    joint_model = load_joint_model(args.joint_model)
    joint_model.to(args.device)

    locomo_qa = load_locomo_qa(args.locomo_path)

    all_utility, all_ttl, all_label = [], [], []
    per_conv_rows = []

    for sample_id, entry in locomo_qa.items():
        answerable_qa = [qa for qa in entry["qa"] if "answer" in qa]
        mids = processed["conversation_to_memory_ids"].get(sample_id)
        if not mids:
            continue
        records = [processed["by_memory_id"][m]["record"] for m in mids]
        embeddings = np.stack([processed["by_memory_id"][m]["embedding"] for m in mids])
        features = np.stack([processed["by_memory_id"][m]["features"] for m in mids])

        objects = build_memory_objects(records, embeddings, features, feature_slices,
                                        survival_model, joint_model, args.device, ttl_quantile=0.5)

        evidence_dia_ids = {d for qa in answerable_qa for d in qa["evidence"]}
        conv_utility, conv_ttl, conv_label = [], [], []
        for o, r in zip(objects, records):
            is_evidence = int(bool(set(_dia_ids_of(r)) & evidence_dia_ids))
            conv_utility.append(o.utility_prob)
            conv_ttl.append(o.predicted_ttl_days)
            conv_label.append(is_evidence)

        all_utility += conv_utility
        all_ttl += conv_ttl
        all_label += conv_label

        n_pos = sum(conv_label)
        if 0 < n_pos < len(conv_label):
            conv_auc_utility = roc_auc_score(conv_label, conv_utility)
            conv_auc_ttl = roc_auc_score(conv_label, conv_ttl)
            per_conv_rows.append({"conversation_id": sample_id, "n_memories": len(conv_label),
                                   "n_evidence": n_pos, "auc_utility": conv_auc_utility, "auc_ttl": conv_auc_ttl})
            print(f"  {sample_id}: n={len(conv_label)}, n_evidence={n_pos}, "
                  f"AUC(utility_prob)={conv_auc_utility:.4f}, AUC(predicted_ttl_days)={conv_auc_ttl:.4f}")
        else:
            print(f"  {sample_id}: skipped per-conversation AUC (n_evidence={n_pos}/{len(conv_label)}, "
                  f"no variation)")

    pooled_n_pos = sum(all_label)
    pooled_auc_utility = roc_auc_score(all_label, all_utility)
    pooled_auc_ttl = roc_auc_score(all_label, all_ttl)

    md_lines = [
        "# Does utility_prob (or predicted_ttl_days) actually predict QA-evidence relevance?",
        "",
        "Free, no-LLM-calls diagnostic: AUC of each signal predicting "
        "`is_evidence` (was this memory ever cited as evidence for a LoCoMo "
        "QA pair in its own conversation), pooled across all LoCoMo memories "
        "and all 10 conversations. 0.5 = random, 1.0 = perfect ranking -- "
        "same scale as the Week-5 Future-Utility head's own validation AUC "
        "(0.71-0.77) and the Week-3/4 survival model's C-index.",
        "",
        f"Pooled: n={len(all_label)} memories, {pooled_n_pos} ({pooled_n_pos/len(all_label):.1%}) are evidence for at least one QA pair.",
        "",
        "| Signal | Pooled AUC (predicts is_evidence) |",
        "|---|---|",
        f"| utility_prob (Future-Utility head) | {pooled_auc_utility:.4f} |",
        f"| predicted_ttl_days (Lifetime head, median quantile) | {pooled_auc_ttl:.4f} |",
        "",
        "## Per-conversation (conversations with both evidence and non-evidence memories only)",
        "",
        "| Conversation | N memories | N evidence | AUC(utility_prob) | AUC(predicted_ttl_days) |",
        "|---|---|---|---|---|",
    ]
    for r in per_conv_rows:
        md_lines.append(f"| {r['conversation_id']} | {r['n_memories']} | {r['n_evidence']} | "
                         f"{r['auc_utility']:.4f} | {r['auc_ttl']:.4f} |")

    md = "\n".join(md_lines)
    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "tables" / "week6_utility_signal_auc.md"
    table_path.write_text(md, encoding="utf-8")
    (out_dir / "raw" / "week6_utility_signal_per_conv.json").write_text(json.dumps(per_conv_rows, indent=2))

    print("\n" + md)
    print(f"\nwritten -> {table_path}")


if __name__ == "__main__":
    main()
