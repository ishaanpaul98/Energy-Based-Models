"""
data/stl10_loader.py  —  BOILERPLATE
All STL-10 DataLoaders for this project.

STL-10 splits used:
  unlabeled       : 100,000 images, no labels  → self-supervised training
  train           :   5,000 labeled images     → linear probe training
  test            :   8,000 labeled images     → linear probe evaluation

torchvision.datasets.STL10 is used directly.
Data is downloaded automatically on first run to cfg.data.root.
"""

from __future__ import annotations
from torch.utils.data import DataLoader
from torchvision.datasets import STL10

from data.transforms import ssl_transform, probe_train_transform, probe_eval_transform


def unlabeled_loader(cfg) -> DataLoader:
    """
    DataLoader for the 100k unlabeled STL-10 images.

    Used as the training set for EBM, I-JEPA, and H-JEPA self-supervised
    training. Labels are not provided by this split.

    Parameters
    ----------
    cfg : EBMConfig | IJEPAConfig | HJEPAConfig
        Any top-level config with a `.data` sub-config (DataConfig).

    Returns
    -------
    DataLoader yielding (images, -1) tuples where
        images : torch.Tensor [B, 3, H, W]  (normalized, ~N(0,1) per channel)
        -1     : dummy label from torchvision (ignore it)

    Note: drop_last=True avoids partial batches, which matter for SGLD
    chain sizing and JEPA masking.
    """
    dataset = STL10(
        root=cfg.data.root,
        split="unlabeled",
        transform=ssl_transform(cfg.data.image_size),
        download=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )


def labeled_train_loader(cfg) -> DataLoader:
    """
    DataLoader for the 5,000 labeled STL-10 training images.
    Used during linear probe training.

    Returns
    -------
    DataLoader yielding (images, labels) tuples where
        labels : torch.LongTensor [B]  class indices in [0, 9]
    """
    dataset = STL10(
        root=cfg.data.root,
        split="train",
        transform=probe_train_transform(cfg.data.image_size),
        download=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )


def labeled_test_loader(cfg) -> DataLoader:
    """
    DataLoader for the 8,000 labeled STL-10 test images.
    Used during linear probe evaluation.

    Returns
    -------
    DataLoader yielding (images, labels) tuples where
        labels : torch.LongTensor [B]  class indices in [0, 9]
    """
    dataset = STL10(
        root=cfg.data.root,
        split="test",
        transform=probe_eval_transform(cfg.data.image_size),
        download=True,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=False,
    )
