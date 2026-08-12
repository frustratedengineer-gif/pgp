"""Save/load for the Week-3 survival model (pycox CoxPH net) and the
Week-5 joint multi-task model."""
import json
from pathlib import Path
import torch
import torchtuples as tt
from pycox.models import CoxPH

from ..heads.survival import build_survival_net
from ..losses.cox_partial import build_cox_model
from .joint_predictor import JointLifecyclePredictor


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


def save_joint_model(model: JointLifecyclePredictor, path: str | Path, config: dict) -> None:
    """Saves weights (.pt) + the architecture config needed to reconstruct
    the module before loading state_dict (embedding_dim/feature_dim/
    fusion_name/hidden sizes aren't recoverable from the state_dict alone)."""
    path = Path(path)
    torch.save(model.state_dict(), path)
    path.with_suffix(".config.json").write_text(json.dumps(config, indent=2))


def load_joint_model(path: str | Path) -> JointLifecyclePredictor:
    path = Path(path)
    config = json.loads(path.with_suffix(".config.json").read_text())
    model = JointLifecyclePredictor(**config)
    # weights_only=True (the default) is fine here: a plain state_dict of
    # tensors, not pycox's nn.Module pickle (contrast load_survival_model above).
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
