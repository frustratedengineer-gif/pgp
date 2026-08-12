"""
Future-utility head (the "Future-utility head: P(retrieved in [t, t+delta])"
box).

Real supervision, restricted to records that actually carry it:
`lifecycle_event == "observed_usage"` (label=1) or `"no_usage_observed"`
(label=0) directly say whether the memory was referenced again later in the
same conversation -- exactly what this head predicts. Records with other
lifecycle_event values (none/update/contradiction/natural_expiry) describe
what happened to the FACT, not whether it was RETRIEVED, so they carry no
direct usage label and are excluded from training (mirrors how
baselines/bucket_classifier.py already restricts its own supervision to the
observed-event subset for the same reason -- see that file's docstring).
The trained head still predicts on every record at inference time; only
fitting is restricted to the labeled subset.
"""
import torch

USAGE_LIFECYCLE_EVENTS = ("observed_usage", "no_usage_observed")


def has_utility_label(lifecycle_event: str) -> bool:
    return lifecycle_event in USAGE_LIFECYCLE_EVENTS


def utility_label_from_lifecycle_event(lifecycle_event: str) -> int:
    if not has_utility_label(lifecycle_event):
        raise ValueError(f"{lifecycle_event!r} carries no direct usage label -- check has_utility_label() first")
    return 1 if lifecycle_event == "observed_usage" else 0


def build_future_utility_net(in_features: int, hidden: int = 128, dropout: float = 0.2) -> torch.nn.Module:
    """MLP -> single logit for P(retrieved again). Consumed by
    losses/utility_loss.py (BCEWithLogitsLoss, so no sigmoid here)."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, 1),
    )
