#!/usr/bin/env python
"""
Downstream memory-policy comparison (baselines/README.md's "not yet
implemented" gap, now implemented): does OUR forgetting policy
(memory/forgetting.py: Lifetime-head TTL expiry + Action-head "forget")
preserve QA-answering quality better than naive alternatives at the SAME
storage budget?

Policies compared: no_forget (ceiling), fifo, lru, ours. For fifo/lru,
capacity is set per-conversation to OUR policy's own natural final active
count -- an apples-to-apples storage-budget comparison at the exact point
our policy actually lands at, not an arbitrarily chosen number.

Simplification (documented, not a shortcut that changes results): only the
FINAL memory-store state is ever queried for QA (LoCoMo has no per-question
timestamp; LongMemEval's question is evaluated once, after its full
haystack). Because TTL-expiry is monotonic in time and action-head/FIFO/LRU
eviction decisions don't depend on prior store state, the final active set
can be computed directly as a set operation instead of simulating a
step-by-step ingest+sweep loop -- verified equivalent to incremental
sweeping for this "final state only" scenario, see module docstring in
memory/forgetting.py for the monotonicity argument. The incremental
store/sweep machinery itself is still exercised for real by
scripts/run_inference_demo.py; this is a legitimate optimization specific
to a final-snapshot-only evaluation, not a reduction in what's tested
elsewhere.

LRU's last_retrieved_at (no live query stream exists during ingestion) is
derived from features.causal.nearest_prior_in_conversation: whenever a
later memory's nearest earlier neighbor is memory X, that's treated as X
being "referenced again" at the later memory's timestamp -- a real,
already-computed signal (the same one behind the novelty/contradiction
features), not fabricated, giving LRU a genuine basis to differ from FIFO.

Coverage caveat: `scripts/build_benchmark.py` measures evidence coverage
(what fraction of each conversation's QA has its evidence memory actually
present in our extracted data) -- ~83.8% mean for LoCoMo. QA pairs whose
evidence was never extracted can't be answered correctly by ANY policy;
this harness does not inflate accuracy by excluding those, so all
policies' accuracy has the same coverage ceiling, not a policy-specific one.

LoCoMo category-5 ("adversarial") exclusion: ~22% of LoCoMo's QA pairs
have no `answer` field at all -- only `adversarial_answer`, which is the
WRONG plausible-sounding decoy a hallucinating system might give; the
correct behavior on these is refusing to answer, not matching it. Scoring
EM/F1 against `adversarial_answer` as if it were a normal reference answer
would reward hallucination and penalize a correct "I don't know," actively
corrupting the comparison. These are excluded from the standard EM/F1
table (`run_locomo` skips any QA entry without an `answer` key) rather than
silently mis-scored.

    OPENROUTER_API_KEY=... python scripts/run_downstream_qa_eval.py \
        --benchmark locomo --device cuda
    OPENROUTER_API_KEY=... python scripts/run_downstream_qa_eval.py \
        --benchmark longmemeval --lme-sample-conversations 100 --lme-splits val test --device cuda
"""
import argparse
import hashlib
import json
import os
import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from memorylife.data.build_benchmark import load_all_processed, load_locomo_qa, load_longmemeval_qa
from memorylife.evaluation.qa_metrics import score_qa
from memorylife.features.causal import nearest_prior_in_conversation
from memorylife.inference.llm_client import chat_completion
from memorylife.inference.pipeline import build_memory_objects, format_retrieved_block
from memorylife.memory.memory_object import MemoryObject
from memorylife.memory.store.numpy_store import NumpyMemoryStore
from memorylife.models.checkpoint import load_joint_model, load_survival_model
from memorylife.retrieval.index import QueryEncoder
from memorylife.retrieval.retriever import Retriever

QA_PROMPT_PATH = Path(__file__).resolve().parent.parent / "src/memorylife/inference/prompts/qa_grounded.txt"


def compute_lru_last_referenced(records, embeddings) -> dict[str, float]:
    prior = nearest_prior_in_conversation(records, embeddings)
    last_referenced: dict[str, float] = {}
    for j, i in prior.items():
        if i is None:
            continue
        touch_time = datetime.fromisoformat(records[j]["injected_at"]).timestamp()
        mid_i = records[i]["memory_id"]
        last_referenced[mid_i] = max(last_referenced.get(mid_i, float("-inf")), touch_time)
    return last_referenced


def remaining_life_fraction(obj: MemoryObject, as_of: datetime) -> float:
    """0 once past predicted_ttl_days, ->1 the fresher a memory is relative
    to its OWN predicted TTL -- a continuous version of `is_expired`'s
    binary check, so it can be combined into a ranking score instead of a
    hard yes/no cutoff."""
    if obj.predicted_ttl_days <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - obj.age_days(as_of) / obj.predicted_ttl_days))


def final_active_ids(objects: list[MemoryObject], policy: str, capacity: int | None,
                      last_referenced: dict[str, float], as_of: datetime) -> set[str]:
    if policy == "no_forget":
        return {o.memory_id for o in objects}
    if policy == "ours":
        return {o.memory_id for o in objects if o.action != "forget" and not o.is_expired(as_of)}
    if policy == "fifo":
        ranked = sorted(objects, key=lambda o: datetime.fromisoformat(o.created_at), reverse=True)
        return {o.memory_id for o in ranked[:capacity]}
    if policy == "lru":
        ranked = sorted(objects, key=lambda o: last_referenced.get(o.memory_id, float("-inf")), reverse=True)
        return {o.memory_id for o in ranked[:capacity]}
    if policy == "ours_utility":
        # `ours` makes an independent per-memory threshold decision; fifo/lru
        # always keep their best-N by a RANKED score. Tests whether ranking by
        # the already-trained (but eviction-unused) Future-Utility head's
        # P(referenced again) -- conceptually the same thing lru approximates
        # heuristically via nearest_prior_in_conversation -- closes the gap.
        ranked = sorted(objects, key=lambda o: o.utility_prob, reverse=True)
        return {o.memory_id for o in ranked[:capacity]}
    if policy == "ours_combo":
        # Blends the trained utility signal with a continuous (not hard-cutoff)
        # version of the lifetime signal, still ranked top-N like fifo/lru.
        ranked = sorted(objects, key=lambda o: 0.5 * o.utility_prob + 0.5 * remaining_life_fraction(o, as_of),
                         reverse=True)
        return {o.memory_id for o in ranked[:capacity]}
    raise ValueError(f"unknown policy: {policy}")


def build_store(objects: list[MemoryObject], active_ids: set[str]) -> NumpyMemoryStore:
    store = NumpyMemoryStore()
    for o in objects:
        if o.memory_id in active_ids:
            store.add(o)
    return store


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.sha256(f"{model}||{prompt}".encode()).hexdigest()


def answer_question(query_encoder, store, prompt_template, question: str, k: int, llm_model: str,
                     cache_dir: Path | None) -> tuple[str, dict]:
    retriever = Retriever(store, query_encoder)
    results = retriever.retrieve(question, k=k)
    prompt = prompt_template.format(retrieved_memories_block=format_retrieved_block(results), query=question)

    if cache_dir is not None:
        cache_path = cache_dir / f"{_cache_key(llm_model, prompt)}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return cached["answer"], cached.get("usage", {})

    api_key = os.environ.get("OPENROUTER_API_KEY")
    text, usage = chat_completion([{"role": "user", "content": prompt}], model=llm_model, api_key=api_key)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"answer": text, "usage": usage}))
    return text, usage


def run_locomo(args, processed, survival_model, joint_model, query_encoder, prompt_template):
    locomo_qa = load_locomo_qa(args.locomo_path)
    rows = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    conv_items = list(locomo_qa.items())
    if args.max_locomo_conversations:
        conv_items = conv_items[: args.max_locomo_conversations]

    n_adversarial_skipped = 0
    for sample_id, entry in conv_items:
        answerable_qa = [qa for qa in entry["qa"] if "answer" in qa]
        n_adversarial_skipped += len(entry["qa"]) - len(answerable_qa)
        entry = {**entry, "qa": answerable_qa}
        if args.max_qa_per_conversation:
            entry = {**entry, "qa": entry["qa"][: args.max_qa_per_conversation]}
        mids = processed["conversation_to_memory_ids"].get(sample_id)
        if not mids:
            print(f"  skipping {sample_id}: no processed memories found")
            continue
        records = [processed["by_memory_id"][m]["record"] for m in mids]
        embeddings = np.stack([processed["by_memory_id"][m]["embedding"] for m in mids])
        features = np.stack([processed["by_memory_id"][m]["features"] for m in mids])

        objects = build_memory_objects(records, embeddings, features, args.feature_slices,
                                              survival_model, joint_model, args.device,
                                              ttl_quantile=args.ttl_quantile)
        as_of = max(datetime.fromisoformat(r["injected_at"]) for r in records)
        last_referenced = compute_lru_last_referenced(records, embeddings)

        ours_ids = final_active_ids(objects, "ours", None, last_referenced, as_of)
        capacity = len(ours_ids)
        print(f"  {sample_id} ({processed['conversation_to_split'][sample_id]}, {len(objects)} memories, "
              f"ours keeps {capacity}):")

        for policy in ("no_forget", "fifo", "lru", "ours", "ours_utility"):
            active_ids = ours_ids if policy == "ours" else final_active_ids(objects, policy, capacity,
                                                                             last_referenced, as_of)
            store = build_store(objects, active_ids)
            cache_dir = Path(args.cache_dir) / "locomo" / policy if args.cache_dir else None

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = [ex.submit(answer_question, query_encoder, store, prompt_template,
                                      qa["question"], args.k, args.llm_model, cache_dir)
                           for qa in entry["qa"]]
                for qa, fut in zip(entry["qa"], futures):
                    answer, usage = fut.result()
                    scores = score_qa(answer, str(qa["answer"]))
                    rows.append({"benchmark": "locomo", "conversation_id": sample_id, "policy": policy,
                                 "store_size": len(active_ids), "em": scores["em"], "f1": scores["f1"],
                                 "question": qa["question"], "reference": str(qa["answer"]), "prediction": answer})
                    for k_, v in usage.items():
                        if k_ in usage_totals:
                            usage_totals[k_] += v
                    usage_totals["calls"] += 1

            policy_rows = [r for r in rows if r["conversation_id"] == sample_id and r["policy"] == policy]
            mean_em = statistics.mean(r["em"] for r in policy_rows)
            mean_f1 = statistics.mean(r["f1"] for r in policy_rows)
            print(f"    {policy:10s} store_size={len(active_ids):4d}  EM={mean_em:.3f}  F1={mean_f1:.3f}")

    print(f"  skipped {n_adversarial_skipped} category-5 (adversarial) QA pairs, see script docstring")
    return rows, usage_totals


def run_longmemeval(args, processed, survival_model, joint_model, query_encoder, prompt_template):
    lme_qa = load_longmemeval_qa(args.longmemeval_path)
    eligible = [cid for cid in processed["conversation_to_memory_ids"]
                if cid in lme_qa and processed["conversation_to_split"][cid] in args.lme_splits]
    rng = random.Random(args.seed)
    sample = rng.sample(eligible, min(args.lme_sample_conversations, len(eligible)))
    print(f"  LongMemEval: {len(eligible)} eligible (splits={args.lme_splits}), sampling {len(sample)}")

    rows = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    for conv_id in sample:
        mids = processed["conversation_to_memory_ids"][conv_id]
        records = [processed["by_memory_id"][m]["record"] for m in mids]
        embeddings = np.stack([processed["by_memory_id"][m]["embedding"] for m in mids])
        features = np.stack([processed["by_memory_id"][m]["features"] for m in mids])
        qa = lme_qa[conv_id]

        objects = build_memory_objects(records, embeddings, features, args.feature_slices,
                                              survival_model, joint_model, args.device,
                                              ttl_quantile=args.ttl_quantile)
        as_of = max(datetime.fromisoformat(r["injected_at"]) for r in records)
        last_referenced = compute_lru_last_referenced(records, embeddings)
        ours_ids = final_active_ids(objects, "ours", None, last_referenced, as_of)
        capacity = len(ours_ids)

        for policy in ("no_forget", "fifo", "lru", "ours", "ours_utility"):
            active_ids = ours_ids if policy == "ours" else final_active_ids(objects, policy, capacity,
                                                                             last_referenced, as_of)
            store = build_store(objects, active_ids)
            cache_dir = Path(args.cache_dir) / "longmemeval" / policy if args.cache_dir else None
            answer, usage = answer_question(query_encoder, store, prompt_template, qa["question"],
                                             args.k, args.llm_model, cache_dir)
            scores = score_qa(answer, str(qa["answer"]))
            rows.append({"benchmark": "longmemeval", "conversation_id": conv_id, "policy": policy,
                         "store_size": len(active_ids), "em": scores["em"], "f1": scores["f1"],
                         "question": qa["question"], "reference": str(qa["answer"]), "prediction": answer})
            for k_, v in usage.items():
                if k_ in usage_totals:
                    usage_totals[k_] += v
            usage_totals["calls"] += 1
        print(f"  {conv_id}: {len(objects)} memories, ours keeps {capacity} -- done")

    return rows, usage_totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["locomo", "longmemeval", "both"], default="both")
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
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lme-sample-conversations", type=int, default=100)
    ap.add_argument("--lme-splits", nargs="+", default=["val", "test"],
                     help="restrict LongMemEval sample to these splits -- val/test conversations were never seen by our trained heads, avoiding train-leakage into the 'ours' policy's evaluation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-locomo-conversations", type=int, default=None,
                     help="debug/pilot: cap how many LoCoMo conversations to run, before committing to the full set")
    ap.add_argument("--max-qa-per-conversation", type=int, default=None,
                     help="debug/pilot: cap QA pairs per LoCoMo conversation")
    ap.add_argument("--ttl-quantile", type=float, default=0.5,
                     help="survival probability S(t) the TTL cutoff crosses; 0.5=median (original Week-6 "
                          "run), lower pushes the cutoff later -- see week6_ttl_quantile_sweep.md")
    ap.add_argument("--out-suffix", default="",
                     help="appended to output filenames so a rerun at a different quantile doesn't "
                          "overwrite the original week6_downstream_qa.md baseline")
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set in environment -- export it and re-run.")

    processed = load_all_processed(args.data_dir, args.emb_dir, args.feat_dir)
    args.feature_slices = json.loads((Path(args.feat_dir) / "feature_slices.json").read_text())

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

    all_rows = []
    all_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}

    if args.benchmark in ("locomo", "both"):
        print("== LoCoMo ==")
        rows, usage = run_locomo(args, processed, survival_model, joint_model, query_encoder, prompt_template)
        all_rows += rows
        for k_ in all_usage:
            all_usage[k_] += usage[k_]

    if args.benchmark in ("longmemeval", "both"):
        print("== LongMemEval ==")
        rows, usage = run_longmemeval(args, processed, survival_model, joint_model, query_encoder, prompt_template)
        all_rows += rows
        for k_ in all_usage:
            all_usage[k_] += usage[k_]

    out_dir = Path(args.out_dir)
    (out_dir / "raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "raw" / f"week6_downstream_qa_raw{args.out_suffix}.json").write_text(json.dumps(all_rows, indent=2))

    summary = {}
    for r in all_rows:
        key = (r["benchmark"], r["policy"])
        summary.setdefault(key, {"em": [], "f1": [], "store_size": []})
        summary[key]["em"].append(r["em"])
        summary[key]["f1"].append(r["f1"])
        summary[key]["store_size"].append(r["store_size"])

    md_lines = ["| Benchmark | Policy | Mean EM | Mean F1 | Mean store size | N questions |", "|---|---|---|---|---|---|"]
    for (benchmark, policy), d in sorted(summary.items()):
        md_lines.append(f"| {benchmark} | {policy} | {statistics.mean(d['em']):.4f} | "
                         f"{statistics.mean(d['f1']):.4f} | {statistics.mean(d['store_size']):.1f} | {len(d['em'])} |")
    table_path = out_dir / "tables" / f"week6_downstream_qa{args.out_suffix}.md"
    table_path.write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))
    print(f"\ntoken usage: {all_usage}")
    print(f"written -> {table_path}")


if __name__ == "__main__":
    main()
