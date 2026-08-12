"""
Action head (the "Action head: store/update/merge/forget" box).

Real, non-fabricated supervision: `lifecycle_event` (already part of the
committed dataset schema, see data/schema.py) is mapped to one of the 4
action classes:
  update        -> "update"   (the memory was superseded by a stated update)
  contradiction -> "merge"    (a later statement conflicts -- the two need
                                reconciling, not a blind overwrite)
  natural_expiry-> "forget"   (the fact simply stopped being true/relevant)
  none / observed_usage / no_usage_observed -> "store"  (default: keep as-is)

This is a genuine label derived from how the dataset's ground truth was
constructed, not a proxy invented for this head -- contrast with
heads/importance.py, which has no such label and is a documented heuristic
instead.
"""
import torch

ACTION_LABELS = ("store", "update", "merge", "forget")

_LIFECYCLE_TO_ACTION = {
    "update": "update",
    "contradiction": "merge",
    "natural_expiry": "forget",
    "none": "store",
    "observed_usage": "store",
    "no_usage_observed": "store",
}


def action_label_from_lifecycle_event(lifecycle_event: str) -> int:
    """Returns the integer class index into ACTION_LABELS."""
    action = _LIFECYCLE_TO_ACTION.get(lifecycle_event)
    if action is None:
        raise ValueError(f"unrecognized lifecycle_event: {lifecycle_event!r}")
    return ACTION_LABELS.index(action)


def build_action_net(in_features: int, hidden: int = 128, dropout: float = 0.2) -> torch.nn.Module:
    """MLP -> logits over ACTION_LABELS. Consumed by losses/action_loss.py."""
    return torch.nn.Sequential(
        torch.nn.Linear(in_features, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden, len(ACTION_LABELS)),
    )
