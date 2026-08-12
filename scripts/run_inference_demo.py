#!/usr/bin/env python
"""Week-5: single-user interactive demo of the full pipeline -- memory
ingestion -> store -> reflection/forgetting/compaction -> retrieval ->
grounded LLM answer -- run against one real conversation from the test
split (default: syn_conv_0108, a synthetic conversation containing
conflicting/updated facts -- e.g. two different phone numbers, two
different favorite colors, two different cities -- specifically so the
demo can show the action head and forgetting/compaction actually doing
something, not just storing everything).

Requires OPENROUTER_API_KEY (see baselines/README.md) for the grounded-QA
LLM call; everything up through retrieval works without it.

    OPENROUTER_API_KEY=... python scripts/run_inference_demo.py
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from memorylife.data.datasets import load_split
from memorylife.encoders.cache import ensure_embeddings
from memorylife.features.pipeline import ensure_features, load_features, slices_path
from memorylife.inference.pipeline import GroundedQAPipeline, build_memory_objects
from memorylife.memory.audit import AuditLog
from memorylife.memory.compaction import find_and_merge_duplicates
from memorylife.memory.forgetting import sweep as forgetting_sweep
from memorylife.memory.store.numpy_store import NumpyMemoryStore
from memorylife.models.checkpoint import load_joint_model, load_survival_model
from memorylife.retrieval.index import QueryEncoder
from memorylife.retrieval.retriever import Retriever

DEFAULT_QUERIES = [
    "Where does the user live?",
    "What is the user's favorite color?",
    "What is the user's phone number?",
    "What does the user do for work?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--emb-dir", default="artifacts/embeddings")
    ap.add_argument("--feat-dir", default="artifacts/features")
    ap.add_argument("--survival-model", default="artifacts/survival_model_net.pt")
    ap.add_argument("--joint-model", default="artifacts/joint_model.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conversation-id", default="syn_conv_0108")
    ap.add_argument("--audit-log", default="artifacts/demo_audit_log.jsonl")
    ap.add_argument("--llm-model", default="openai/gpt-4o")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    args = ap.parse_args()

    ensure_embeddings(args.data_dir, args.emb_dir, [args.split, "train"], device=args.device)
    ensure_features(args.data_dir, args.emb_dir, args.feat_dir, [args.split], device=args.device)

    split = load_split(args.data_dir, args.emb_dir, args.split)
    features = load_features(args.feat_dir, args.split)
    slices = json.loads(slices_path(args.feat_dir).read_text())

    keep_idx = [i for i, r in enumerate(split["records"]) if r["conversation_id"] == args.conversation_id]
    if not keep_idx:
        raise SystemExit(f"no records found for conversation_id={args.conversation_id!r} in split={args.split!r}")
    records = [split["records"][i] for i in keep_idx]
    embeddings = split["embeddings"][keep_idx]
    conv_features = features[keep_idx]

    survival_model = load_survival_model(args.survival_model, embeddings.shape[1])
    # checkpoints don't reliably persist baseline hazards -- recompute from
    # the same training split used to fit the model (see also
    # scripts/compute_richer_metrics.py, which needs the exact same fix)
    train_split = load_split(args.data_dir, args.emb_dir, "train")
    survival_model.compute_baseline_hazards(
        input=train_split["embeddings"].astype("float32"),
        target=(train_split["durations"], train_split["events"]),
    )
    joint_model = load_joint_model(args.joint_model)

    print(f"== Ingesting {len(records)} memories from conversation {args.conversation_id} ==")
    objects = build_memory_objects(records, embeddings, conv_features, slices,
                                    survival_model, joint_model, device="cpu")

    store = NumpyMemoryStore()
    for obj in objects:
        store.add(obj)
        print(f"  {obj.memory_id}: action={obj.action:8s} importance={obj.importance:.2f} "
              f"ttl={obj.predicted_ttl_days:7.1f}d utility={obj.utility_prob:.2f} | {obj.text[:60]}")

    audit_log = AuditLog(args.audit_log)

    print("\n== Compaction (merging near-duplicate memories) ==")
    merged = find_and_merge_duplicates(store, audit_log)
    print(f"  merged {len(merged)} pair(s): {merged}" if merged else "  no near-duplicates found")

    print("\n== Forgetting sweep (action-head 'forget' + TTL-expired) ==")
    # as_of=None here would use real wall-clock time, which is meaningless
    # against dataset timestamps from 2023-2026 -- use the latest timestamp
    # in this conversation as "now" instead, so age_days is meaningful
    as_of = max(datetime.fromisoformat(r["injected_at"]) for r in records)
    forgotten = forgetting_sweep(store, audit_log, as_of=as_of)
    print(f"  forgot {len(forgotten)} memor{'y' if len(forgotten)==1 else 'ies'}: {forgotten}"
          if forgotten else "  nothing forgotten yet at this point in the conversation")

    print(f"\n== Active memories remaining: {len(store.all())} / {len(objects)} ==")
    print(f"Audit log: {args.audit_log} ({len(audit_log.read())} entries)")

    print("\n== Grounded QA (retrieval + LLM) ==")
    query_encoder = QueryEncoder(device="cpu")
    retriever = Retriever(store, query_encoder)
    pipeline = GroundedQAPipeline(retriever, llm_model=args.llm_model)

    for query in args.queries:
        result = pipeline.answer(query, k=3)
        print(f"\nQ: {query}")
        print(f"A: {result['answer']}")
        print("   grounded in:")
        for m in result["retrieved_memories"]:
            print(f"     - ({m['score']:.2f}) {m['text']}")


if __name__ == "__main__":
    main()
