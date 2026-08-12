"""Dispatch a fusion module by name -- keeps configs declarative
(fusion: concat | gated) instead of scripts importing a specific class."""
from .concat import ConcatFusion
from .gated import GatedFusion

REGISTRY = {
    "concat": ConcatFusion,
    "gated": GatedFusion,
}


def build_fusion(name: str, embedding_dim: int, feature_dim: int):
    if name not in REGISTRY:
        raise ValueError(f"unknown fusion '{name}', choices: {sorted(REGISTRY)}")
    return REGISTRY[name](embedding_dim, feature_dim)
