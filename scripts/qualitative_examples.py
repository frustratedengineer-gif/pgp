#!/usr/bin/env python
"""
Week-6 reviewer gap: every claim so far is a number in a table. This pulls
concrete, traceable examples for the write-up -- no new LLM calls (reuses
the already-collected predictions in
results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json and
results/raw/week6_judge_scores_..._pilot.json), plus one free local
inference pass (already-trained checkpoints, no training) to recover
which evidence memory was evicted under which policy.

Two categories:
1. "Smoking gun" eviction examples: a question whose gold-evidence memory
   was evicted under `ours` (TTL threshold) but survived under
   `ours_utility` (ranked eviction) -- AND ours's answer was actually
   wrong while ours_utility's was judged correct. Traces the full causal
   chain: evicted memory text -> wrong answer -> fixed by keeping it.
2. EM-vs-judge disagreements: already surfaced in week6_judge_scores*.md,
   re-selected here for the clearest, most illustrative cases (short,
   unambiguous paraphrases) rather than the first 20 in file order.

    python scripts/qualitative_examples.py --device cuda
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from memorylife.data.build_benchmark import _dia_ids_of, load_all_processed, load_locomo_qa
from memorylife.inference.pipeline import build_memory_objects
from memorylife.models.checkpoint import load_joint_model, load_survival_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_downstream_qa_eval import compute_lru_last_referenced, final_active_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--feat-dir", default="artifacts/features")
    ap.add_argument("--locomo-path", default="data/raw/locomo10.json")
    ap.add_argument("--survival-model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--joint-model", default="artifacts/joint_model.pt")
    ap.add_argument("--ttl-quantile", type=float, default=0.2, help="must match the pilot run being cross-referenced")
    ap.add_argument("--predictions", default="results/raw/week6_downstream_qa_raw_q0.2_ranked_pilot.json")
    ap.add_argument("--judged", default="results/raw/week6_judge_scores_week6_downstream_qa_raw_q0.2_ranked_pilot.json")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-examples", type=int, default=8)
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
    predictions = json.loads(Path(args.predictions).read_text())
    judged = json.loads(Path(args.judged).read_text())

    # question -> {policy: {em, f1, prediction, judge}}
    pred_by_q: dict[tuple, dict] = {}
    for r in predictions:
        pred_by_q.setdefault((r["benchmark"], r["conversation_id"], r["question"]), {})[r["policy"]] = dict(r)
    for r in judged:
        key = (r["benchmark"], r["conversation_id"], r["question"])
        if key in pred_by_q and r["policy"] in pred_by_q[key]:
            pred_by_q[key][r["policy"]]["judge"] = r["judge"]

    smoking_gun = []
    for sample_id, entry in locomo_qa.items():
        answerable_qa = [qa for qa in entry["qa"] if "answer" in qa]
        mids = processed["conversation_to_memory_ids"].get(sample_id)
        if not mids:
            continue
        records = [processed["by_memory_id"][m]["record"] for m in mids]
        embeddings = np.stack([processed["by_memory_id"][m]["embedding"] for m in mids])
        features = np.stack([processed["by_memory_id"][m]["features"] for m in mids])

        objects = build_memory_objects(records, embeddings, features, feature_slices,
                                        survival_model, joint_model, args.device, ttl_quantile=args.ttl_quantile)
        objects_by_id = {o.memory_id: o for o in objects}
        as_of = max(datetime.fromisoformat(r["injected_at"]) for r in records)
        last_referenced = compute_lru_last_referenced(records, embeddings)

        ours_ids = final_active_ids(objects, "ours", None, last_referenced, as_of)
        capacity = len(ours_ids)
        utility_ids = final_active_ids(objects, "ours_utility", capacity, last_referenced, as_of)

        dia_to_mids: dict[str, list[str]] = {}
        for m in mids:
            for d in _dia_ids_of(processed["by_memory_id"][m]["record"]):
                dia_to_mids.setdefault(d, []).append(m)

        for qa in answerable_qa:
            key = ("locomo", sample_id, qa["question"])
            if key not in pred_by_q or "ours" not in pred_by_q[key] or "ours_utility" not in pred_by_q[key]:
                continue  # not in the pilot's question subset
            evidence_mids = {m for d in qa["evidence"] for m in dia_to_mids.get(d, [])}
            if not evidence_mids:
                continue
            evicted_under_ours = evidence_mids - ours_ids
            survived_under_utility = evidence_mids & utility_ids
            if not (evicted_under_ours and survived_under_utility):
                continue

            ours_row = pred_by_q[key]["ours"]
            util_row = pred_by_q[key]["ours_utility"]
            ours_wrong = ours_row.get("judge", ours_row["em"]) == 0
            util_right = util_row.get("judge", util_row["em"]) == 1
            if not (ours_wrong and util_right):
                continue

            evidence_texts = [objects_by_id[m].text for m in evicted_under_ours]
            smoking_gun.append({
                "conversation_id": sample_id, "question": qa["question"], "reference": str(qa["answer"]),
                "evicted_evidence_text": evidence_texts,
                "ours_prediction": ours_row["prediction"], "ours_em": ours_row["em"], "ours_judge": ours_row.get("judge"),
                "ours_utility_prediction": util_row["prediction"], "ours_utility_em": util_row["em"],
                "ours_utility_judge": util_row.get("judge"),
            })

    smoking_gun = smoking_gun[: args.max_examples]

    seen = set()
    disagreements = []
    for policies in pred_by_q.values():
        for r in policies.values():
            if r["em"] == 0.0 and r.get("judge") == 1.0:
                dedupe_key = (r["question"], str(r["reference"]), r["prediction"])
                if dedupe_key not in seen:
                    seen.add(dedupe_key)
                    disagreements.append(r)
    # prefer short, clean, unambiguous-looking disagreements for the write-up
    disagreements.sort(key=lambda r: len(r["prediction"]) + len(str(r["reference"])))
    disagreements = disagreements[: args.max_examples]

    md_lines = [
        "# Week-6 qualitative examples",
        "",
        "## 1. Traced eviction failures: `ours` evicts the evidence and gets it wrong; `ours_utility` keeps it and gets it right",
        "",
        f"{len(smoking_gun)} example(s) found where the SAME evidence memory was evicted under the "
        "original TTL-threshold policy but survived under utility-ranked eviction, AND that specific "
        "change flipped the answer from wrong to right (judged, not just EM).",
        "",
    ]
    for i, ex in enumerate(smoking_gun, 1):
        md_lines += [
            f"### Example {i} ({ex['conversation_id']})",
            f"**Question:** {ex['question']}",
            f"**Reference answer:** {ex['reference']}",
            f"**Evicted evidence memory (present for `ours_utility`, gone for `ours`):** "
            + "; ".join(f'"{t}"' for t in ex["evicted_evidence_text"]),
            f"**`ours` (TTL threshold) answered:** \"{ex['ours_prediction']}\" -- WRONG "
            f"(EM={ex['ours_em']}, judge={ex['ours_judge']})",
            f"**`ours_utility` (ranked) answered:** \"{ex['ours_utility_prediction']}\" -- CORRECT "
            f"(EM={ex['ours_utility_em']}, judge={ex['ours_utility_judge']})",
            "",
        ]
    if not smoking_gun:
        md_lines.append("(none found in this question sample -- see script docstring for the exact "
                         "criteria; try a larger sample or a different quantile.)")

    md_lines += [
        "",
        "## 2. EM penalizes correct-but-differently-worded answers",
        "",
        "Shortest, cleanest examples where EM=0 but the LLM judge scored the answer substantively "
        "correct (selected for clarity, not cherry-picked for a particular policy).",
        "",
        "**Caveat -- not blindly trustworthy:** the judge prompt (`prompts/judge.txt`) does not see "
        "memory dates, only question/reference/prediction. A duration-vs-absolute-date pair (e.g. "
        "reference \"three years\" vs. prediction \"2019\") can only be judged correct if the LLM "
        "silently infers the reference date -- plausibly a JUDGE ERROR, not proof EM was too harsh. "
        "Inspect entries like this manually before quoting them; most of the table is not this kind, "
        "but not all of it is guaranteed clean.",
        "",
        "| Question | Reference | Prediction | Why EM=0 despite being correct |",
        "|---|---|---|---|",
    ]
    for r in disagreements:
        md_lines.append(f"| {r['question']} | {r['reference']} | {r['prediction']} | "
                         f"format/wording differs, meaning matches |")

    md = "\n".join(md_lines)
    out_dir = Path(args.out_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "week6_qualitative_examples.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwritten -> {out_path}")


if __name__ == "__main__":
    main()
