"""Save/load for the Week-3 survival model (pycox CoxPH net)."""
from pathlib import Path
import torch
import torchtuples as tt
from pycox.models import CoxPH

from ..heads.survival import build_survival_net
from ..losses.cox_partial import build_cox_model


def save_survival_model(model: CoxPH, path: str | Path) -> None:
    model.save_net(str(path))


def load_survival_model(path: str | Path, in_features: int) -> CoxPH:
    net = build_survival_net(in_features)
    model = build_cox_model(net)
    # weights_only=False: this is our own checkpoint, not a third-party file.
    # PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects
    # pycox's plain nn.Module pickle.
    model.load_net(str(path), weights_only=False)
    return model
