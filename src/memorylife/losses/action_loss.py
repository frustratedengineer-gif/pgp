"""Class-weighted cross-entropy for the action head. The 4 action classes
are heavily imbalanced (in the Week-3/4 train split: store is the large
majority, forget/merge/update are minorities -- see
data/event_labeling.py-derived lifecycle_event counts), so an unweighted CE
would mostly just learn to always predict "store". Weights are inverse
class frequency, computed from the labels actually passed in (not
hardcoded), matching the class_weight="balanced" convention already used
for bucket_classifier (baselines/bucket_classifier.py).
"""
import numpy as np
import torch


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)  # avoid div-by-zero for an absent class
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_action_loss(labels: np.ndarray, n_classes: int) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss(weight=class_weights(labels, n_classes))
