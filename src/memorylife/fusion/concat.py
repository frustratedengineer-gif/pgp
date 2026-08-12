"""Fusion baseline: concatenate embedding + auxiliary features. No learned
parameters -- the "does fusion help at all" ablation baseline against gated.py.
"""
import torch


class ConcatFusion(torch.nn.Module):
    def __init__(self, embedding_dim: int, feature_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim

    @property
    def out_dim(self) -> int:
        return self.embedding_dim + self.feature_dim

    def forward(self, embedding: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return torch.cat([embedding, features], dim=-1)
