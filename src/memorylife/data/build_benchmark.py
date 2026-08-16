"""
Loads and links the raw LoCoMo / LongMemEval benchmark files (QA pairs +
dialogue text) to this repo's already-extracted, already-labeled memory
records (`data/processed/*_survival.jsonl`) -- for
`scripts/run_downstream_qa_eval.py`'s downstream memory-system comparison.

This is NOT the original dialogue -> candidate-memory extraction pipeline
that produced `data/raw/*.jsonl` in the first place (that pipeline is still
not in this repo, see `data/README.md`'s "Known gap"). This is a narrower
job: read the ALREADY-PUBLISHED QA pairs from the raw benchmark files and
link them to conversation_id, which was already confirmed to match
LoCoMo's `sample_id` and `lme_<question_id>` for LongMemEval by direct
lookup before this file was written (all 10 LoCoMo and all 500 LongMemEval
conversation_ids are present in data/processed/*.jsonl).

Also merges records/embeddings/features across all 3 processed splits into
one conversation_id-indexed lookup, since a LoCoMo/LongMemEval conversation
can land in any split and the downstream eval needs to fetch it regardless
of which one.
"""
import json
from pathlib import Path

from .datasets import load_split
from ..features.pipeline import load_features


def load_locomo_qa(path: str | Path) -> dict[str, dict]:
    """Returns {sample_id: {"qa": [...], "conversation": {...}}}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {c["sample_id"]: {"qa": c["qa"], "conversation": c["conversation"]} for c in data}


def load_longmemeval_qa(path: str | Path) -> dict[str, dict]:
    """Returns {f"lme_{question_id}": {question, answer, question_date, ...}}."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {f"lme_{d['question_id']}": d for d in data}


def load_all_processed(data_dir: str | Path, emb_dir: str | Path, feat_dir: str | Path,
                        splits: tuple[str, ...] = ("train", "val", "test")) -> dict:
    """Merges records/embeddings/features across splits. Returns:
        {
          "by_memory_id": {memory_id: {"record": ..., "embedding": ..., "features": ..., "split": ...}},
          "conversation_to_memory_ids": {conversation_id: [memory_id, ...]},
          "conversation_to_split": {conversation_id: split_name},
        }
    """
    by_memory_id = {}
    conversation_to_memory_ids: dict[str, list[str]] = {}
    conversation_to_split: dict[str, str] = {}

    for split_name in splits:
        split_data = load_split(data_dir, emb_dir, split_name)
        feats = load_features(feat_dir, split_name)
        for i, mid in enumerate(split_data["ids"]):
            record = split_data["records"][i]
            by_memory_id[mid] = {
                "record": record, "embedding": split_data["embeddings"][i],
                "features": feats[i], "split": split_name,
            }
            conv_id = record["conversation_id"]
            conversation_to_memory_ids.setdefault(conv_id, []).append(mid)
            conversation_to_split[conv_id] = split_name

    return {
        "by_memory_id": by_memory_id,
        "conversation_to_memory_ids": conversation_to_memory_ids,
        "conversation_to_split": conversation_to_split,
    }


def _dia_ids_of(record: dict) -> list[str]:
    """evidence_dia_id is usually a single string but is a list for a
    handful of records with multi-turn evidence (confirmed: 7/1929 LoCoMo
    records in the train split) -- normalize to a flat list either way."""
    val = record.get("evidence_dia_id")
    if val is None:
        return []
    return val if isinstance(val, list) else [val]


def evidence_coverage(locomo_qa: dict, processed: dict) -> dict[str, float]:
    """What fraction of each LoCoMo conversation's QA pairs have at least
    one evidence dia_id present among our extracted memories? A ceiling on
    achievable downstream QA accuracy independent of forgetting policy --
    evidence never extracted can't be answered by any policy. Real,
    measured number (see docs/reproducibility.md), not assumed 100%."""
    coverage = {}
    for sample_id, entry in locomo_qa.items():
        mids = processed["conversation_to_memory_ids"].get(sample_id, [])
        our_dia_ids = {d for m in mids for d in _dia_ids_of(processed["by_memory_id"][m]["record"])}
        total = len(entry["qa"])
        if total == 0:
            continue
        covered = sum(1 for qa in entry["qa"] if any(e in our_dia_ids for e in qa["evidence"]))
        coverage[sample_id] = covered / total
    return coverage
