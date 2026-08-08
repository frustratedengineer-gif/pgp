"""
Assembles one split's labeled records + cached embeddings into aligned
numpy arrays for the survival model / baselines / evaluation.

A plain function returning arrays, not a torch.utils.data.Dataset: nothing
downstream (pycox's CoxPH.fit, sklearn's LogisticRegression) needs batched
tensor loading, so a Dataset/DataLoader would be unused ceremony. If a
future head needs minibatch streaming, wrap this in a thin
torch.utils.data.Dataset here rather than changing the callers.
"""
import numpy as np

from ..encoders.cache import load_embeddings
from ..utils.io import load_jsonl


def load_split(processed_dir: str, embeddings_dir: str, split: str) -> dict:
    records = load_jsonl(f"{processed_dir}/{split}_survival.jsonl")
    rec_by_id = {r["memory_id"]: r for r in records}

    ids, emb = load_embeddings(embeddings_dir, split)
    assert len(ids) == len(records), f"{split}: embedding/record count mismatch"

    ordered_records = [rec_by_id[i] for i in ids]
    durations = np.array([r["duration_days"] for r in ordered_records], dtype=np.float32)
    events = np.array([r["event_observed"] for r in ordered_records], dtype=np.int32)

    return {
        "records": ordered_records,
        "ids": ids,
        "embeddings": emb,
        "durations": durations,
        "events": events,
    }
