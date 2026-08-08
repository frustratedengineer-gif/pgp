"""
Lifetime head (the "Lifetime head (survival)" box: hazard h(t|z) -> S(t|z), TTL).

Week 3 scope is the survival head alone, trained directly on the fused
embedding z (no other heads/features fused in yet -- that's the joint
4-head model in `models/joint_predictor.py`, not built yet).
"""
import torch


def build_survival_net(in_features: int) -> torch.nn.Module:
    """MLP -> single log partial-hazard score. Consumed by
    pycox.models.CoxPH (see losses/cox_partial.py)."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(256),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(64, 1),
    )
