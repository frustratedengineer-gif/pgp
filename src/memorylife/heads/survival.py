"""
Lifetime head (the "Lifetime head (survival)" box: hazard h(t|z) -> S(t|z), TTL).

Week 3 scope is the survival head alone, trained directly on the fused
embedding z (no other heads/features fused in yet -- that's the joint
4-head model in `models/joint_predictor.py`, not built yet).
"""
import torch


def build_survival_net(in_features: int, hidden1: int = 256, hidden2: int = 64,
                        dropout1: float = 0.2, dropout2: float = 0.1) -> torch.nn.Module:
    """MLP -> single log partial-hazard score. Consumed by
    pycox.models.CoxPH (see losses/cox_partial.py). Defaults match the
    Week-3 architecture; hidden1/hidden2/dropout1/dropout2 are exposed for
    the Week-4 hyperparameter-sensitivity ablation (scripts/run_ablations.py)."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(hidden1),
        torch.nn.Dropout(dropout1),
        torch.nn.Linear(hidden1, hidden2),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout2),
        torch.nn.Linear(hidden2, 1),
    )
