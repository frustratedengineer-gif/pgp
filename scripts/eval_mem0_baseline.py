#!/usr/bin/env python
"""
Reviewer gap #1: a real memory-system baseline (Mem0), not just our own
eviction policies compared against each other -- see
baselines/mem0_wrapper.py's module docstring for the local-LLM
substitution, the timestamp limitation, and the disclosed confound of
using a weaker extraction model than Mem0's own typical evaluation setup.

Checkpointed / resumable by design: indexing writes to a persistent
Qdrant path (not deleted between runs) and a JSON checkpoint tracks which
conversation_ids are already fully indexed, so a killed or deliberately
paused run picks back up without re-indexing anything already done. Same
invocation every time.

    python scripts/eval_mem0_baseline.py --device cuda:7 \
        --local-llm-url http://127.0.0.1:8420/v1
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baselines"))
from mem0_wrapper import add_turn, build_memory, search  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_downstream_qa_eval import QA_PROMPT_PATH, _cache_key  # noqa: E402

from memorylife.evaluation.qa_metrics import score_qa  # noqa: E402
from memorylife.inference.llm_client import chat_completion  # noqa: E402


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"indexed_conversations": []}


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def answer_with_mem0(memories: list[str], question: str, prompt_template: str, llm_model: str,
                      cache_dir: Path) -> tuple[str, dict]:
    block = "\n".join(f"- {m}" for m in memories) if memories else "(no memories retrieved)"
    prompt = prompt_template.format(retrieved_memories_block=block, query=question)
    cache_path = cache_dir / f"{_cache_key(llm_model, prompt)}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return cached["answer"], cached.get("usage", {})
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")
    text, usage = chat_completion([{"role": "user", "content": prompt}], model=llm_model, api_key=api_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"answer": text, "usage": usage}))
    return text, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo-path", default="data/raw/locomo10.json")
    ap.add_argument("--qdrant-path", default="artifacts/mem0_qdrant")
    ap.add_argument("--checkpoint", default="artifacts/mem0_checkpoint.json")
    ap.add_argument("--cache-dir", default="artifacts/llm_cache/mem0_baseline")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--llm-model", default="openai/gpt-4o", help="answering model (paid, small cost)")
    ap.add_argument("--local-llm-url", default="http://127.0.0.1:8420/v1", help="Mem0's own indexing LLM (free)")
    ap.add_argument("--use-local-llm", action="store_true", default=True)
    ap.add_argument("--max-qa-per-conversation", type=int, default=12, help="matches the ranked pilot's sample")
    ap.add_argument("--conversations", nargs="+", default=None,
                     help="restrict to these conversation IDs (default: all 10)")
    args = ap.parse_args()

    import os
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set -- needed for the (small, paid) answering step.")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    data = json.loads(Path(args.locomo_path).read_text(encoding="utf-8"))
    if args.conversations:
        data = [c for c in data if c["sample_id"] in args.conversations]

    mem = build_memory("locomo_mem0_eval", args.qdrant_path, use_local_llm=args.use_local_llm,
                        local_llm_url=args.local_llm_url)
    prompt_template = QA_PROMPT_PATH.read_text()
    cache_dir = Path(args.cache_dir)

    rows = []
    raw_path = Path(args.out_dir) / "raw" / "week6_mem0_baseline_raw.json"
    if raw_path.exists():
        rows = json.loads(raw_path.read_text(encoding="utf-8"))

    for c in data:
        sample_id = c["sample_id"]
        if sample_id in checkpoint["indexed_conversations"]:
            print(f"  {sample_id}: already indexed, skipping indexing (checkpoint)")
        else:
            conv = c["conversation"]
            session_keys = sorted((k for k in conv if k.startswith("session_") and not k.endswith("date_time")),
                                  key=lambda k: int(k.split("_")[1]))
            t0 = time.time()
            n_turns = 0
            for sess_key in session_keys:
                date_str = conv.get(f"{sess_key}_date_time", "")
                for turn in conv[sess_key]:
                    add_turn(mem, sample_id, turn["speaker"], turn["text"], f"[{date_str}]")
                    n_turns += 1
                    if n_turns % 50 == 0:
                        elapsed = time.time() - t0
                        print(f"    {sample_id}: {n_turns} turns indexed ({elapsed:.0f}s, "
                              f"{elapsed/n_turns:.2f}s/turn)")
            print(f"  {sample_id}: indexed {n_turns} turns in {time.time()-t0:.0f}s")
            checkpoint["indexed_conversations"].append(sample_id)
            save_checkpoint(checkpoint_path, checkpoint)

        # --- answer this conversation's QA sample (small, paid; cached, so reruns are free) ---
        already_answered = {(r["conversation_id"], r["question"]) for r in rows}
        answerable_qa = [qa for qa in c["qa"] if "answer" in qa][: args.max_qa_per_conversation]
        for qa in answerable_qa:
            if (sample_id, qa["question"]) in already_answered:
                continue
            memories = search(mem, sample_id, qa["question"], top_k=10)
            answer, usage = answer_with_mem0(memories, qa["question"], prompt_template, args.llm_model, cache_dir)
            scores = score_qa(answer, str(qa["answer"]))
            rows.append({"benchmark": "locomo", "conversation_id": sample_id, "policy": "mem0",
                         "n_memories_retrieved": len(memories), "question": qa["question"],
                         "reference": str(qa["answer"]), "prediction": answer, **scores})
        Path(args.out_dir, "raw").mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(rows, indent=2))
        print(f"  {sample_id}: {len(answerable_qa)} questions answered/cached")

    mean_em = statistics.mean(r["em"] for r in rows) if rows else float("nan")
    mean_f1 = statistics.mean(r["f1"] for r in rows) if rows else float("nan")
    mean_bleu1 = statistics.mean(r["bleu1"] for r in rows) if rows else float("nan")
    md = (f"# Mem0 baseline (reviewer gap #1)\n\n"
          f"See baselines/mem0_wrapper.py for the local-LLM substitution and disclosed limitations.\n\n"
          f"| Policy | N | Mean EM | Mean F1 | Mean BLEU-1 |\n|---|---|---|---|---|\n"
          f"| mem0 | {len(rows)} | {mean_em:.4f} | {mean_f1:.4f} | {mean_bleu1:.4f} |\n")
    out_path = Path(args.out_dir) / "tables" / "week6_mem0_baseline.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"written -> {out_path}")
    print(f"conversations indexed so far: {checkpoint['indexed_conversations']}")


if __name__ == "__main__":
    main()
