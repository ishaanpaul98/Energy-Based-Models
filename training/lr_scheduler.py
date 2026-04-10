"""
training/lr_scheduler.py  —  BOILERPLATE
Cosine decay with linear warmup — the standard schedule for ViT-based SSL.
"""

from __future__ import annotations
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Returns a LambdaLR that:
      - linearly ramps lr from 0 -> base_lr over `warmup_epochs`
      - cosine-decays lr from base_lr -> min_lr over the remaining epochs

    Parameters
    ----------
    optimizer      : PyTorch optimizer whose base lr is already set.
    warmup_epochs  : number of linear warmup epochs.
    total_epochs   : total training epochs.
    min_lr_ratio   : min_lr as a fraction of base_lr (default 0 = decay to 0).

    Returns
    -------
    LambdaLR scheduler — call scheduler.step() once per epoch.

    Example
    -------
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    >>> scheduler = cosine_warmup_scheduler(optimizer, warmup_epochs=40, total_epochs=300)
    >>> for epoch in range(300):
    ...     train_one_epoch(...)
    ...     scheduler.step()
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

    return LambdaLR(optimizer, lr_lambda)
