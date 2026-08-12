"""Learned gated fusion (the "Feature Fusion -> fused vector z" box, "ours"
per the architecture figure's colour legend). Projects the auxiliary
feature block into the embedding space, then learns a per-dimension gate
deciding how much of the projected features to blend in vs. trusting the
raw embedding alone -- lets the model down-weight noisy/irrelevant
extractor outputs per-example instead of always concatenating them at full
weight (concat.py's fixed alternative).

Output dim == embedding_dim (unlike ConcatFusion), so it drops in wherever
a head expects the encoder's native width.
"""
import torch


class GatedFusion(torch.nn.Module):
    def __init__(self, embedding_dim: int, feature_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.project = torch.nn.Linear(feature_dim, embedding_dim)
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, embedding_dim),
            torch.nn.Sigmoid(),
        )

    @property
    def out_dim(self) -> int:
        return self.embedding_dim

    def forward(self, embedding: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        proj_features = self.project(features)
        g = self.gate(torch.cat([embedding, proj_features], dim=-1))
        return embedding * (1 - g) + proj_features * g
