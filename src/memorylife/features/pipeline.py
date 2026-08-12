"""Runs every feature extractor over a split and returns a fixed-order,
fixed-width feature matrix + the per-extractor slice boundaries (so a
fusion/ablation script can drop one extractor's columns without touching
the others). Cached to disk per split, same convention as
encoders/cache.py's embedding cache, since re-running 4 HF pipelines over
thousands of records on every train invocation would be wasteful.
"""
from pathlib import Path

import numpy as np

from .base import FeatureExtractor
from .contradiction import ContradictionFeatures
from .emotion import EmotionFeatures
from .entities import EntityFeatures
from .intent import IntentFeatures
from .novelty import NoveltyFeatures
from .temporal import TemporalFeatures

# fixed order -- the fused vector's feature block layout depends on this
EXTRACTOR_ORDER = ("temporal", "novelty", "entities", "intent", "emotion", "contradiction")


def build_extractors(device: str = "cpu") -> dict[str, FeatureExtractor]:
    return {
        "temporal": TemporalFeatures(),
        "novelty": NoveltyFeatures(),
        "entities": EntityFeatures(device=device),
        "intent": IntentFeatures(device=device),
        "emotion": EmotionFeatures(device=device),
        "contradiction": ContradictionFeatures(device=device),
    }


def feature_slices(extractors: dict[str, FeatureExtractor]) -> dict[str, tuple[int, int]]:
    """name -> (start_col, end_col) in the concatenated matrix, in EXTRACTOR_ORDER."""
    slices = {}
    col = 0
    for name in EXTRACTOR_ORDER:
        dim = extractors[name].dim
        slices[name] = (col, col + dim)
        col += dim
    return slices


def features_path(cache_dir: str | Path, split: str) -> Path:
    return Path(cache_dir) / f"{split}_features.npy"


def slices_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "feature_slices.json"


def compute_features(records: list[dict], embeddings: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, dict]:
    extractors = build_extractors(device=device)
    slices = feature_slices(extractors)
    blocks = [extractors[name].extract(records, embeddings=embeddings) for name in EXTRACTOR_ORDER]
    return np.concatenate(blocks, axis=1).astype(np.float32), slices


def ensure_features(data_dir: str | Path, emb_dir: str | Path, cache_dir: str | Path,
                     splits: list[str], device: str = "cpu") -> None:
    """Compute+cache any of `splits` not already cached. Mirrors
    encoders.cache.ensure_embeddings."""
    import json

    from ..data.datasets import load_split

    missing = [s for s in splits if not features_path(cache_dir, s).exists()]
    if not missing:
        return

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    slices_out = None
    for split in missing:
        split_data = load_split(data_dir, emb_dir, split)
        feats, slices = compute_features(split_data["records"], split_data["embeddings"], device=device)
        np.save(features_path(cache_dir, split), feats)
        slices_out = slices
    if slices_out is not None and not slices_path(cache_dir).exists():
        slices_path(cache_dir).write_text(json.dumps(slices_out, indent=2))


def load_features(cache_dir: str | Path, split: str) -> np.ndarray:
    return np.load(features_path(cache_dir, split))
