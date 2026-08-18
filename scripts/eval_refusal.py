#!/usr/bin/env python
"""
Reviewer gap (from comparing against REMem, ICLR 2026 -- see
paper/draft.md's companion gap-analysis): scripts/run_downstream_qa_eval.py
EXCLUDES LoCoMo's category-5 ("adversarial") QA pairs entirely -- there is
no `answer` field, only `adversarial_answer`, a plausible-sounding WRONG
decoy. Excluding them avoids scoring EM/F1 against a decoy (still correct,
see that script's docstring), but it also means we never measure whether
eviction-induced information loss makes a policy MORE likely to
hallucinate a confident wrong answer instead of honestly refusing.

This measures exactly that, using the same methodology as REMem's Table 6
(refusal precision/recall/F1): for the SAME 10 LoCoMo conversations and
the SAME per-conversation QA subset already scored in
week6_downstream_qa_raw_q0.2_ranked_pilot.json (so those calls are 100%
cache hits, free), add an equal-sized sample of that conversation's
category-5 adversarial questions (new LLM calls -- the only real cost of
this script) and classify every answer (`memorylife.evaluation.qa_metrics.
is_refusal`) as a refusal or not. Ground truth: adversarial questions are
the "truly unanswerable" set.

    precision = TP / (TP + FP)   -- of predicted refusals, how many were genuinely unanswerable
    recall    = TP / (TP + FN)   -- of genuinely unanswerable questions, how many were refused
    F1        = harmonic mean

A low recall means the policy is HALLUCINATING against the adversarial
decoy instead of refusing -- directly relevant to whether an eviction
policy that discards evidence also discards the system's ability to know
it no longer knows something.

    OPENROUTER_API_KEY=... python scripts/eval_refusal.py --device cuda
"""
import argparse
import json
import random
import statistics
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from memorylife.data.build_benchmark import load_all_processed, load_locomo_qa
from memorylife.evaluation.qa_metrics import is_refusal, score_qa
from memorylife.inference.pipeline import build_memory_objects
from memorylife.models.checkpoint import load_joint_model, load_survival_model
from memorylife.retrieval.index import QueryEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_downstream_qa_eval import (  # noqa: E402
    QA_PROMPT_PATH, answer_question, build_store, compute_lru_last_referenced, final_active_ids,
)

POLICIES = ("no_forget", "fifo", "lru", "ours", "ours_utility")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--feat-dir", default="artifacts/features")
    ap.add_argument("--locomo-path", default="data/raw/locomo10.json")
    ap.add_argument("--survival-model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--joint-model", default="artifacts/joint_model.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache-dir", default="artifacts/llm_cache/downstream_qa",
                     help="same default as run_downstream_qa_eval.py so the non-adversarial questions "
                          "from the q0.2 ranked pilot are cache hits, not re-paid-for")
    ap.add_argument("--llm-model", default="openai/gpt-4o")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ttl-quantile", type=float, default=0.2,
                     help="must match the pilot being extended (week6_downstream_qa_raw_q0.2_ranked_pilot.json)")
    ap.add_argument("--max-qa-per-conversation", type=int, default=12,
                     help="matches the ranked pilot's non-adversarial cap, for a balanced sample")
    ap.add_argument("--max-adversarial-per-conversation", type=int, default=12,
                     help="cap on NEW adversarial questions per conversation -- this is the real cost")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import os
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set in environment -- export it and re-run.")

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
    query_encoder = QueryEncoder(device="cpu")

    prompt_template = QA_PROMPT_PATH.read_text()
    locomo_qa = load_locomo_qa(args.locomo_path)
    rng = random.Random(args.seed)

    rows = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    for sample_id, entry in locomo_qa.items():
        mids = processed["conversation_to_memory_ids"].get(sample_id)
        if not mids:
            continue
        answerable_qa = [qa for qa in entry["qa"] if "answer" in qa][: args.max_qa_per_conversation]
        adversarial_qa = [qa for qa in entry["qa"] if qa.get("category") == 5]
        rng.shuffle(adversarial_qa)
        adversarial_qa = adversarial_qa[: args.max_adversarial_per_conversation]
        qa_pool = [(qa, False) for qa in answerable_qa] + [(qa, True) for qa in adversarial_qa]

        records = [processed["by_memory_id"][m]["record"] for m in mids]
        embeddings = np.stack([processed["by_memory_id"][m]["embedding"] for m in mids])
        features = np.stack([processed["by_memory_id"][m]["features"] for m in mids])

        objects = build_memory_objects(records, embeddings, features, feature_slices,
                                        survival_model, joint_model, args.device, ttl_quantile=args.ttl_quantile)
        as_of = max(datetime.fromisoformat(r["injected_at"]) for r in records)
        last_referenced = compute_lru_last_referenced(records, embeddings)
        ours_ids = final_active_ids(objects, "ours", None, last_referenced, as_of)
        capacity = len(ours_ids)
        print(f"  {sample_id}: {len(answerable_qa)} answerable + {len(adversarial_qa)} adversarial "
              f"questions, ours keeps {capacity}/{len(objects)}")

        for policy in POLICIES:
            active_ids = ours_ids if policy == "ours" else final_active_ids(objects, policy, capacity,
                                                                              last_referenced, as_of)
            store = build_store(objects, active_ids)
            cache_dir = Path(args.cache_dir) / "locomo" / policy

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(answer_question, query_encoder, store, prompt_template,
                                      qa["question"], args.k, args.llm_model, cache_dir)
                           for qa, _ in qa_pool]
                for (qa, is_adversarial), fut in zip(qa_pool, futures):
                    answer, usage = fut.result()
                    refusal = is_refusal(answer)
                    row = {"conversation_id": sample_id, "policy": policy, "question": qa["question"],
                           "prediction": answer, "is_adversarial": is_adversarial, "is_refusal": refusal}
                    if not is_adversarial:
                        row.update(score_qa(answer, str(qa["answer"])))
                    rows.append(row)
                    for k_, v in usage.items():
                        if k_ in usage_totals:
                            usage_totals[k_] += v
                    usage_totals["calls"] += 1

    # --- refusal precision/recall/F1, pooled per policy (matches REMem Table 6's definition) ---
    summary = {}
    for policy in POLICIES:
        policy_rows = [r for r in rows if r["policy"] == policy]
        tp = sum(1 for r in policy_rows if r["is_refusal"] and r["is_adversarial"])
        fp = sum(1 for r in policy_rows if r["is_refusal"] and not r["is_adversarial"])
        fn = sum(1 for r in policy_rows if not r["is_refusal"] and r["is_adversarial"])
        n_refusals = tp + fp
        n_adversarial = sum(1 for r in policy_rows if r["is_adversarial"])
        precision = tp / n_refusals if n_refusals else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        if precision == precision and recall == recall and (precision + recall) > 0:  # not NaN, not 0/0
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")
        summary[policy] = {"n_refusals": n_refusals, "n_adversarial": n_adversarial,
                            "precision": precision, "recall": recall, "f1": f1}

    md_lines = [
        "# Refusal behavior on LoCoMo's adversarial (category-5) questions",
        "",
        "Same methodology as REMem (ICLR 2026) Table 6: precision/recall/F1 of correctly refusing "
        "genuinely unanswerable questions, pooled across all 10 LoCoMo conversations at matched "
        "storage budget per policy. Ground truth unanswerable = category-5 adversarial QA pairs "
        "(no real `answer` field, only a plausible-sounding wrong `adversarial_answer` decoy).",
        "",
        f"N adversarial questions per policy: {summary[POLICIES[0]]['n_adversarial']}. "
        f"N answerable (non-adversarial) questions per policy: "
        f"{sum(1 for r in rows if r['policy'] == POLICIES[0] and not r['is_adversarial'])}.",
        "",
        "| Policy | # Refusals | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    for policy in POLICIES:
        s = summary[policy]
        md_lines.append(f"| {policy} | {s['n_refusals']} | {s['precision']:.3f} | "
                         f"{s['recall']:.3f} | {s['f1']:.3f} |")

    md_lines += ["", f"Token usage (includes cache hits at face value): {usage_totals}", ""]

    md = "\n".join(md_lines)
    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "tables" / "week6_refusal_eval.md"
    table_path.write_text(md, encoding="utf-8")
    (out_dir / "raw" / "week6_refusal_eval_raw.json").write_text(json.dumps(rows, indent=2))

    print("\n" + md)
    print(f"\ntoken usage: {usage_totals}")
    print(f"written -> {table_path}")


if __name__ == "__main__":
    main()
