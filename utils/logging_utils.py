"""
utils/logging_utils.py  —  BOILERPLATE
TensorBoard writer factory + convenience helpers used by all training scripts.
"""

from __future__ import annotations
import os
from torch.utils.tensorboard import SummaryWriter


def make_writer(log_dir: str, run_name: str) -> SummaryWriter:
    """
    Create a TensorBoard SummaryWriter under log_dir/run_name.
    The directory is created if it does not exist.

    Parameters
    ----------
    log_dir  : str  base log directory from cfg.log.log_dir
    run_name : str  human-readable identifier, e.g. "ebm_run1"

    Returns
    -------
    SummaryWriter

    Example
    -------
    >>> writer = make_writer("./runs/ebm", "experiment_1")
    >>> # then: tensorboard --logdir ./runs/ebm
    """
    path = os.path.join(log_dir, run_name)
    os.makedirs(path, exist_ok=True)
    return SummaryWriter(log_dir=path)


def log_scalars(
    writer: SummaryWriter,
    tag_value_dict: dict[str, float],
    step: int,
) -> None:
    """
    Write multiple scalars in one call.

    Parameters
    ----------
    writer         : active SummaryWriter
    tag_value_dict : e.g. {"loss/train": 0.42, "loss/neg": 1.1}
    step           : global training step
    """
    for tag, value in tag_value_dict.items():
        writer.add_scalar(tag, value, global_step=step)


def log_images(
    writer: SummaryWriter,
    tag: str,
    images,        # torch.Tensor [B, C, H, W] in [0, 1]
    step: int,
    nrow: int = 8,
) -> None:
    """
    Write a grid of images to TensorBoard.

    Parameters
    ----------
    writer : active SummaryWriter
    tag    : display name in TensorBoard (e.g. "samples/sgld")
    images : torch.Tensor [B, C, H, W], values expected in [0, 1]
    step   : global training step
    nrow   : images per row in the grid
    """
    import torchvision.utils as vutils
    grid = vutils.make_grid(images[:nrow].clamp(0, 1), nrow=nrow, normalize=False)
    writer.add_image(tag, grid, global_step=step)
