"""
training/ema.py  —  BOILERPLATE
Exponential Moving Average update for target encoder parameters.
Used by both I-JEPA and H-JEPA.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


@torch.no_grad()
def update_ema(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    momentum: float,
) -> None:
    """
    In-place EMA update:  theta_target <- m * theta_target + (1 - m) * theta_online

    Parameters
    ----------
    online_encoder : the context encoder being trained with gradients.
    target_encoder : the frozen (no_grad) target encoder.
    momentum       : EMA coefficient (e.g. 0.996). Higher = slower update.

    Usage
    -----
    Called once per training step AFTER the optimizer.step():
        momentum = cosine_ema_schedule(...)
        update_ema(context_enc, target_enc, momentum)
    """
    for online_p, target_p in zip(
        online_encoder.parameters(), target_encoder.parameters()
    ):
        target_p.data.mul_(momentum).add_(online_p.data, alpha=1.0 - momentum)


def cosine_ema_schedule(
    base_momentum: float,
    final_momentum: float,
    current_step: int,
    total_steps: int,
) -> float:
    """
    Cosine annealing of EMA momentum from base_momentum -> final_momentum.
    Mirrors the schedule used in I-JEPA (Appendix A).

    Parameters
    ----------
    base_momentum  : starting EMA value (e.g. 0.996)
    final_momentum : ending EMA value   (e.g. 1.0)
    current_step   : current global training step (0-indexed)
    total_steps    : total number of training steps

    Returns
    -------
    float : momentum value for this step

    Example
    -------
    >>> total = num_epochs * steps_per_epoch
    >>> for step, (x, _) in enumerate(loader):
    ...     m = cosine_ema_schedule(0.996, 1.0, step, total)
    ...     update_ema(context_enc, target_enc, m)
    """
    return final_momentum - (final_momentum - base_momentum) * (
        math.cos(math.pi * current_step / total_steps) + 1.0
    ) / 2.0
