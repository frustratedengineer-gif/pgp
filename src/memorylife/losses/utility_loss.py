"""Binary cross-entropy for the future-utility head, computed only over
records with a genuine usage label (see heads/future_utility.py). Positive
weight compensates for the observed_usage/no_usage_observed class
imbalance (~1:2 in the Week-3/4 train split), same inverse-frequency
convention as losses/action_loss.py."""
import numpy as np
import torch


def pos_weight(labels: np.ndarray) -> torch.Tensor:
    n_pos = max(int(labels.sum()), 1)
    n_neg = max(len(labels) - n_pos, 1)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def build_utility_loss(labels: np.ndarray) -> torch.nn.Module:
    return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight(labels))
