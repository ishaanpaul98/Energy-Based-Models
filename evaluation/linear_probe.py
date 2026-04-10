"""
evaluation/linear_probe.py  —  STUB
Frozen-backbone linear probe evaluation.

A linear probe is the standard evaluation protocol for self-supervised
representations: freeze the pretrained encoder, train a single linear layer on
top of it using the labeled data, and report top-1 accuracy on the test set.

A high linear probe accuracy means the encoder has learned linearly separable
features — a strong indicator of representation quality without fine-tuning.

Reference
---------
The linear probe protocol is standard in SSL literature; see e.g. Chen et al.
(2020) SimCLR, He et al. (2022) MAE, and Assran et al. (2023) I-JEPA for
how results are reported.

Design notes
------------
- Freeze the encoder: `for p in encoder.parameters(): p.requires_grad = False`
- Use the mean-pooled patch tokens as the image representation:
    z = encoder(x)              # [B, N, D]  (keep_ids=None → all patches)
    z = z.mean(dim=1)           # [B, D]     mean-pool over patch tokens
- Train only the linear head on labeled STL-10 train split (5k images).
- Evaluate on labeled STL-10 test split (8k images).
- 10 classes, so head is nn.Linear(embed_dim, 10).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LinearProbe(nn.Module):
    """
    Linear classifier head on top of a frozen encoder.

    Parameters
    ----------
    embed_dim  : int  dimension of the encoder's output tokens
    num_classes : int  number of target classes (10 for STL-10)
    """

    def __init__(self, embed_dim: int, num_classes: int = 10) -> None:
        super().__init__()

        # TODO: define self.head = nn.Linear(embed_dim, num_classes)
        # Optionally add a nn.BatchNorm1d(embed_dim, affine=False) BEFORE the
        # linear layer — this is common practice for linear probes and stabilises
        # training when representations have varying scales.

        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : [B, embed_dim]  mean-pooled patch representations from the encoder

        Returns
        -------
        logits : [B, num_classes]
        """
        # TODO: return self.head(z)
        raise NotImplementedError


def train_linear_probe(
    encoder: nn.Module,
    probe: LinearProbe,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    n_epochs: int = 100,
) -> None:
    """
    Train the linear probe on the labeled training split.
    The encoder is frozen — only probe parameters receive gradients.

    Parameters
    ----------
    encoder      : pretrained encoder (VisionTransformer or EnergyNet)
    probe        : LinearProbe  (trainable)
    train_loader : DataLoader   yields (images, labels) from labeled_train_loader
    optimizer    : optimiser over probe.parameters() ONLY (not encoder)
    device       : torch.device
    n_epochs     : number of training epochs (typically 100)

    Implementation:
    ---------------
    encoder.eval()                           # freeze BN/Dropout if any
    for param in encoder.parameters():       # belt-and-suspenders freeze
        param.requires_grad_(False)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        probe.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Extract frozen representations
            with torch.no_grad():
                z = encoder(images)          # [B, N, D]  (ViT) or [B, D] (EBM)
                if z.dim() == 3:
                    z = z.mean(dim=1)        # mean-pool patch tokens -> [B, D]

            # Forward + backward on probe only
            logits = probe(z)
            loss   = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    """
    raise NotImplementedError


def evaluate_linear_probe(
    encoder: nn.Module,
    probe: LinearProbe,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate top-1 accuracy on the test split.

    Parameters
    ----------
    encoder     : pretrained encoder (frozen)
    probe       : trained LinearProbe
    test_loader : DataLoader  yields (images, labels) from labeled_test_loader
    device      : torch.device

    Returns
    -------
    accuracy : float  top-1 accuracy in [0, 1]

    Implementation:
    ---------------
    encoder.eval()
    probe.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            z = encoder(images)
            if z.dim() == 3:
                z = z.mean(dim=1)
            logits  = probe(z)
            preds   = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total
    """
    raise NotImplementedError
