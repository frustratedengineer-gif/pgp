"""
Joint Lifecycle Predictor (the architecture figure's central box): encoder
embedding + auxiliary features -> fusion -> 3 heads trained jointly
(Lifetime/survival, Action, Future-utility) sharing one fused representation
z. Importance is deliberately NOT a 4th trained head here -- see
heads/importance.py for why (no ground-truth label exists).

Embedding and features are passed in precomputed (from
encoders/cache.py + features/pipeline.py), not computed inline -- keeps
this module a pure nn.Module trainable with a standard optimizer loop,
consistent with how heads/survival.py's net is used in Week 3/4.
"""
import torch

from ..fusion.build import build_fusion
from ..heads.action import ACTION_LABELS, build_action_net
from ..heads.future_utility import build_future_utility_net
from ..heads.survival import build_survival_net


class JointLifecyclePredictor(torch.nn.Module):
    def __init__(self, embedding_dim: int, feature_dim: int, fusion_name: str = "gated",
                 survival_hidden1: int = 256, survival_hidden2: int = 64,
                 action_hidden: int = 128, utility_hidden: int = 128):
        super().__init__()
        self.fusion = build_fusion(fusion_name, embedding_dim, feature_dim)
        z_dim = self.fusion.out_dim
        self.survival_net = build_survival_net(z_dim, hidden1=survival_hidden1, hidden2=survival_hidden2)
        self.action_net = build_action_net(z_dim, hidden=action_hidden)
        self.utility_net = build_future_utility_net(z_dim, hidden=utility_hidden)

    def forward(self, embedding: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.fusion(embedding, features)
        return {
            "log_hazard": self.survival_net(z).squeeze(-1),
            "action_logits": self.action_net(z),
            "utility_logit": self.utility_net(z).squeeze(-1),
        }


__all__ = ["JointLifecyclePredictor", "ACTION_LABELS"]
