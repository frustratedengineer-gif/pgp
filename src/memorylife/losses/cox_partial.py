"""
Cox partial-likelihood loss, censoring-aware.

We use pycox's `CoxPH` model (Cox partial log-likelihood, correctly handling
right-censored (T, delta) pairs) rather than reimplementing it -- this
module is a thin, named wrapper so the loss has an explicit home in the
package the way docs/... describes it, and so a from-scratch reference
version can be swapped in later without touching callers.
"""
import torch
import torchtuples as tt
from pycox.models import CoxPH


def build_cox_model(net: torch.nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4) -> CoxPH:
    optimizer = tt.optim.Adam(lr=lr, weight_decay=weight_decay)
    return CoxPH(net, optimizer)
