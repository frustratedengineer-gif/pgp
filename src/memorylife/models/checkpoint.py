"""Save/load for the Week-3 survival model (pycox CoxPH net)."""
from pathlib import Path
import torch
import torchtuples as tt
from pycox.models import CoxPH

from ..heads.survival import build_survival_net
from ..losses.cox_partial import build_cox_model


def save_survival_model(model: CoxPH, path: str | Path) -> None:
    model.save_net(str(path))


def load_survival_model(path: str | Path, in_features: int, hidden1: int = 256, hidden2: int = 64,
                         dropout1: float = 0.2, dropout2: float = 0.1) -> CoxPH:
    """hidden1/hidden2/dropout1/dropout2 must match what the checkpoint was
    trained with (defaults match Week 3) -- needed to load a Week-4
    hyperparameter-ablation checkpoint, since the architecture isn't stored
    in the .pt file itself, only the weights."""
    net = build_survival_net(in_features, hidden1=hidden1, hidden2=hidden2,
                              dropout1=dropout1, dropout2=dropout2)
    model = build_cox_model(net)
    # weights_only=False: this is our own checkpoint, not a third-party file.
    # PyTorch >=2.6 defaults torch.load to weights_only=True, which rejects
    # pycox's plain nn.Module pickle.
    model.load_net(str(path), weights_only=False)
    return model
