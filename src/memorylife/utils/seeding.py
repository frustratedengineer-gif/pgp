"""Deterministic seeding across random/numpy/torch. Week 3 results were run
with seed=42 only; experiments/seeds.txt lists the 5 seeds a full paper run
should average over (not yet done -- see docs/reproducibility.md)."""
import os
import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
