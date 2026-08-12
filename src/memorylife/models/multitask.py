"""
Multi-task loss combination for JointLifecyclePredictor: weighted sum of
the 3 real heads' losses, with the utility term masked to only the records
that carry a genuine usage label (see heads/future_utility.py).

Fixed scalar weights (not learned/uncertainty-weighted -- that's listed as
a future refinement, not required for a first joint-model checkpoint).
Cox partial-likelihood uses pycox's own standalone loss function
(pycox.models.loss.cox_ph_loss) rather than the CoxPH wrapper class, since
the wrapper drives its own training loop internally and can't share
gradients with the action/utility heads through one fusion backbone --
this is why Week 5's joint model has its own custom training loop
(scripts/train_joint.py) instead of reusing losses/cox_partial.py's
build_cox_model wrapper from Week 3.
"""
from dataclasses import dataclass

import torch
from pycox.models.loss import cox_ph_loss


@dataclass
class LossWeights:
    survival: float = 1.0
    action: float = 0.5
    utility: float = 0.5


def compute_joint_loss(
    outputs: dict[str, torch.Tensor],
    durations: torch.Tensor,
    events: torch.Tensor,
    action_labels: torch.Tensor,
    utility_labels: torch.Tensor,
    utility_mask: torch.Tensor,
    action_loss_fn: torch.nn.Module,
    utility_loss_fn: torch.nn.Module,
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, float]]:
    survival_loss = cox_ph_loss(outputs["log_hazard"], durations, events)
    action_loss = action_loss_fn(outputs["action_logits"], action_labels)

    if utility_mask.any():
        utility_loss = utility_loss_fn(outputs["utility_logit"][utility_mask], utility_labels[utility_mask])
    else:
        utility_loss = torch.tensor(0.0, device=outputs["utility_logit"].device)

    total = weights.survival * survival_loss + weights.action * action_loss + weights.utility * utility_loss
    breakdown = {
        "survival_loss": float(survival_loss.detach()),
        "action_loss": float(action_loss.detach()),
        "utility_loss": float(utility_loss.detach()),
        "total_loss": float(total.detach()),
    }
    return total, breakdown
