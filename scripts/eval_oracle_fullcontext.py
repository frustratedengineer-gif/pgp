#!/usr/bin/env python
"""
Reviewer gap (comparing against REMem, ICLR 2026, which always anchors
its results between "Oracle Message" -- given gold evidence only -- and
"Full-Context" -- the entire corpus in the prompt): our `no_forget`
"ceiling" retains every memory but STILL goes through top-k=5 retrieval,
so it conflates two different costs: "how much does eviction cost us"
and "how much does limiting the prompt to the top-5 retrieved memories
cost us, even with nothing evicted." This adds the two missing reference
points so those two costs can be told apart.

- **oracle**: store built from ONLY a QA pair's own gold-evidence
  memories (the same dia_to_mids evidence-linkage machinery
  diagnose_eviction_evidence.py and qualitative_examples.py already use),
  retrieved with k covering all of them. Isolates "how good is the
  LLM's answering, given PERFECT retrieval and PERFECT eviction" -- the
  true theoretical ceiling no forgetting policy could ever exceed on its
  own, since even handing the model exactly the right memory can still
  fail on parsing/reasoning/format grounds (see the error taxonomy).
- **full_context**: store built from ALL of the conversation's memories
  (same set as `no_forget`), retrieved with k = store size so nothing is
  dropped by the usual top-5 cutoff. Isolates whether top-k retrieval
  itself, independent of any eviction policy, is already costing
  accuracy.

Scope, to control real API cost: reuses the SAME LoCoMo 120-question /
LongMemEval 25-question pilot sample as
week6_downstream_qa_q0.2_ranked_pilot.md for `oracle` (cheap -- an
evidence-only store is tiny, usually 1-3 memories). `full_context` is
scoped down further on LoCoMo specifically
(--locomo-fullcontext-cap-per-conv, default 3 questions/conversation =
30 total) since LoCoMo stores are large (~185-260 memories per
conversation, week6_downstream_qa.md) and EVERY one of them goes in the
prompt for this policy -- expensive per call. LongMemEval gets the full
25-question `full_context` sample regardless, since its haystacks are
small (~7 memories, llm_token_usage.md-scale) even before any eviction.

    OPENROUTER_API_KEY=... python scripts/eval_oracle_fullcontext.py --device cuda
"""
import argparse
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from memorylife.data.build_benchmark import (_dia_ids_of, load_all_processed, load_locomo_qa,
                                              load_longmemeval_qa)
from memorylife.evaluation.qa_metrics import score_qa
from memorylife.inference.pipeline import build_memory_objects
from memorylife.models.checkpoint import load_joint_model, load_survival_model
from memorylife.retrieval.index import QueryEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_downstream_qa_eval import QA_PROMPT_PATH, answer_question, build_store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--feat-dir", default="artifacts/features")
    ap.add_argument("--locomo-path", default="data/raw/locomo10.json")
    ap.add_argument("--longmemeval-path", default="data/raw/longmemeval_s_cleaned.json")
    ap.add_argument("--survival-model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--joint-model", default="artifacts/joint_model.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--cache-dir", default="artifacts/llm_cache/downstream_qa")
    ap.add_argument("--llm-model", default="openai/gpt-4o")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ttl-quantile", type=float, default=0.2, help="unused by oracle/full_context construction, kept for build_memory_objects' signature")
    ap.add_argument("--max-qa-per-conversation", type=int, default=12, help="matches the ranked pilot's sample")
    ap.add_argument("--locomo-fullcontext-cap-per-conv", type=int, default=3,
                     help="cost control: full_context prompts on LoCoMo are large (whole store), so only "
                          "the first N answerable questions per conversation get a full_context call")
    ap.add_argument("--lme-sample-conversations", type=int, default=25)
    ap.add_argument("--lme-splits", nargs="+", default=["val", "test"])
    ap.add_argument("--seed", type=int, default=42, help="must match the pilot for the same LongMemEval sample")
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

    rows = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    def answer_and_score(benchmark, conv_id, question, reference, store, k, policy):
        cache_dir = Path(args.cache_dir) / benchmark / policy
        answer, usage = answer_question(query_encoder, store, prompt_template, question, k, args.llm_model, cache_dir)
        row = {"benchmark": benchmark, "conversation_id": conv_id, "policy": policy, "store_size": len(store.all()),
               "question": question, "reference": str(reference), "prediction": answer}
        row.update(score_qa(answer, str(reference)))
        for k_, v in usage.items():
            if k_ in usage_totals:
                usage_totals[k_] += v
        usage_totals["calls"] += 1
        return row

    # --- LoCoMo ---
    print("== LoCoMo ==")
    locomo_qa = load_locomo_qa(args.locomo_path)
    for sample_id, entry in locomo_qa.items():
        mids = processed["conversation_to_memory_ids"].get(sample_id)
        if not mids:
            continue
        answerable_qa = [qa for qa in entry["qa"] if "answer" in qa][: args.max_qa_per_conversation]
        records = [processed["by_memory_id"][m]["record"] for m in mids]
        embeddings = np.stack([processed["by_memory_id"][m]["embedding"] for m in mids])
        features = np.stack([processed["by_memory_id"][m]["features"] for m in mids])
        objects = build_memory_objects(records, embeddings, features, feature_slices,
                                        survival_model, joint_model, args.device, ttl_quantile=args.ttl_quantile)
        objects_by_id = {o.memory_id: o for o in objects}
        dia_to_mids: dict[str, list[str]] = {}
        for m in mids:
            for d in _dia_ids_of(processed["by_memory_id"][m]["record"]):
                dia_to_mids.setdefault(d, []).append(m)

        full_store = build_store(objects, {o.memory_id for o in objects})
        n_fullcontext = 0

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = []
            for qa in answerable_qa:
                evidence_mids = {m for d in qa["evidence"] for m in dia_to_mids.get(d, [])}
                if evidence_mids:
                    oracle_store = build_store(objects, evidence_mids)
                    futures.append(ex.submit(answer_and_score, "locomo", sample_id, qa["question"], qa["answer"],
                                              oracle_store, max(len(evidence_mids), 1), "oracle"))
                if n_fullcontext < args.locomo_fullcontext_cap_per_conv:
                    futures.append(ex.submit(answer_and_score, "locomo", sample_id, qa["question"], qa["answer"],
                                              full_store, len(objects), "full_context"))
                    n_fullcontext += 1
            for fut in futures:
                rows.append(fut.result())
        print(f"  {sample_id}: {len(objects)} memories -- oracle + full_context done")

    # --- LongMemEval ---
    print("\n== LongMemEval ==")
    lme_qa = load_longmemeval_qa(args.longmemeval_path)
    eligible = [cid for cid in processed["conversation_to_memory_ids"]
                if cid in lme_qa and processed["conversation_to_split"][cid] in args.lme_splits]
    rng = random.Random(args.seed)
    sample = rng.sample(eligible, min(args.lme_sample_conversations, len(eligible)))
    print(f"  {len(sample)} conversations sampled (matches the ranked pilot's seed/splits)")

    for conv_id in sample:
        mids = processed["conversation_to_memory_ids"][conv_id]
        records = [processed["by_memory_id"][m]["record"] for m in mids]
        embeddings = np.stack([processed["by_memory_id"][m]["embedding"] for m in mids])
        features = np.stack([processed["by_memory_id"][m]["features"] for m in mids])
        qa = lme_qa[conv_id]
        objects = build_memory_objects(records, embeddings, features, feature_slices,
                                        survival_model, joint_model, args.device, ttl_quantile=args.ttl_quantile)
        dia_to_mids: dict[str, list[str]] = {}
        for m in mids:
            for d in _dia_ids_of(processed["by_memory_id"][m]["record"]):
                dia_to_mids.setdefault(d, []).append(m)

        evidence_mids = {m for d in qa["answer_session_ids"] for m in dia_to_mids.get(d, [])}
        full_store = build_store(objects, {o.memory_id for o in objects})

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = []
            if evidence_mids:
                oracle_store = build_store(objects, evidence_mids)
                futures.append(ex.submit(answer_and_score, "longmemeval", conv_id, qa["question"], qa["answer"],
                                          oracle_store, max(len(evidence_mids), 1), "oracle"))
            futures.append(ex.submit(answer_and_score, "longmemeval", conv_id, qa["question"], qa["answer"],
                                      full_store, len(objects), "full_context"))
            for fut in futures:
                rows.append(fut.result())
        print(f"  {conv_id}: {len(objects)} memories -- done")

    # --- summarize ---
    summary: dict[tuple, dict] = {}
    for r in rows:
        key = (r["benchmark"], r["policy"])
        summary.setdefault(key, {"em": [], "f1": [], "bleu1": []})
        summary[key]["em"].append(r["em"])
        summary[key]["f1"].append(r["f1"])
        summary[key]["bleu1"].append(r["bleu1"])

    md_lines = [
        "# Oracle and Full-Context reference points (reviewer gap)",
        "",
        "Same methodology as REMem (ICLR 2026): `oracle` = answer given ONLY the QA pair's own gold-"
        "evidence memories (no eviction, no retrieval noise -- the true ceiling); `full_context` = "
        "answer given the ENTIRE conversation's memory store, uncapped by the usual top-5 retrieval "
        "(isolates retrieval-k cost from eviction cost). Compare against `no_forget` in "
        "week6_downstream_qa_q0.2_ranked_pilot.md, which retains everything but still goes through "
        "top-5 retrieval.",
        "",
        "| Benchmark | Policy | N | Mean EM | Mean F1 | Mean BLEU-1 |",
        "|---|---|---|---|---|---|",
    ]
    for (benchmark, policy), d in sorted(summary.items()):
        md_lines.append(f"| {benchmark} | {policy} | {len(d['em'])} | {statistics.mean(d['em']):.4f} | "
                         f"{statistics.mean(d['f1']):.4f} | {statistics.mean(d['bleu1']):.4f} |")

    md_lines += ["", f"Token usage (includes cache hits at face value): {usage_totals}", ""]

    md = "\n".join(md_lines)
    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "tables" / "week6_oracle_fullcontext.md"
    table_path.write_text(md, encoding="utf-8")
    (out_dir / "raw" / "week6_oracle_fullcontext_raw.json").write_text(json.dumps(rows, indent=2))

    print("\n" + md)
    print(f"\ntoken usage: {usage_totals}")
    print(f"written -> {table_path}")


if __name__ == "__main__":
    main()
