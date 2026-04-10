"""
data/transforms.py  —  BOILERPLATE
Augmentation presets for STL-10 (96x96 images).

Three presets:
  ssl_transform         : strong augmentation for self-supervised training
  probe_train_transform : moderate augmentation for linear probe training
  probe_eval_transform  : deterministic eval transform
"""

from __future__ import annotations
from torchvision import transforms

# Per-channel mean/std computed on STL-10
_MEAN = (0.4467, 0.4398, 0.4066)
_STD  = (0.2603, 0.2566, 0.2713)


def ssl_transform(image_size: int = 96) -> transforms.Compose:
    """
    Strong augmentation for self-supervised training (EBM / I-JEPA / H-JEPA).

    Returns a single view; masking provides the context/target split at the
    patch level for JEPA-style models. Follows augmentation conventions from
    the I-JEPA paper (Assran et al. 2023, Appendix A).
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.4, hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def probe_train_transform(image_size: int = 96) -> transforms.Compose:
    """Light augmentation for the labeled training split during linear probe."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])


def probe_eval_transform(image_size: int = 96) -> transforms.Compose:
    """Deterministic center-crop for validation/test during linear probe."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
