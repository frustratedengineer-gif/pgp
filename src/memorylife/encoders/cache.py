"""On-disk embedding cache: one .npy + one .txt (memory_id order) per split."""
from pathlib import Path
import numpy as np

from .base import SentenceEncoder


def embeddings_path(cache_dir: str | Path, split: str) -> Path:
    return Path(cache_dir) / f"{split}_emb.npy"


def ids_path(cache_dir: str | Path, split: str) -> Path:
    return Path(cache_dir) / f"{split}_ids.txt"


def save_embeddings(cache_dir: str | Path, split: str, ids: list[str], emb: np.ndarray) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path(cache_dir, split), emb.astype(np.float32))
    ids_path(cache_dir, split).write_text("\n".join(ids), encoding="utf-8")


def load_embeddings(cache_dir: str | Path, split: str) -> tuple[list[str], np.ndarray]:
    ids = ids_path(cache_dir, split).read_text(encoding="utf-8").splitlines()
    emb = np.load(embeddings_path(cache_dir, split))
    return ids, emb


def encode_split(encoder: SentenceEncoder, records: list[dict], cache_dir: str | Path, split: str,
                  batch_size: int = 128) -> tuple[list[str], np.ndarray]:
    """Encode records (using their `memory_id`/`text` fields) and cache to disk."""
    ids = [r["memory_id"] for r in records]
    texts = [r["text"] for r in records]
    emb = encoder.encode(texts, batch_size=batch_size)
    save_embeddings(cache_dir, split, ids, emb)
    return ids, emb


def ensure_embeddings(data_dir: str | Path, cache_dir: str | Path, splits: list[str],
                       device: str = "cuda", model_name: str | None = None) -> None:
    """Encode+cache any of `splits` not already cached. Every caller that
    needs a split's embeddings (train.py, baselines, evaluate.py) should
    call this first instead of assuming some other script already ran --
    that assumption is what broke bucket_classifier on the test split."""
    missing = [s for s in splits if not embeddings_path(cache_dir, s).exists()]
    if not missing:
        return

    # imported lazily so callers that only ever hit the cache-hit path
    # don't need sentence-transformers/torch importable
    from .bge import BGEEncoder
    from ..utils.io import load_jsonl

    kwargs = {"device": device}
    if model_name:
        kwargs["model_name"] = model_name
    encoder = BGEEncoder(**kwargs)

    for split in missing:
        records = load_jsonl(Path(data_dir) / f"{split}_survival.jsonl")
        encode_split(encoder, records, cache_dir, split)
