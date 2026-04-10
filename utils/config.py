"""
utils/config.py  —  BOILERPLATE
YAML -> Python dataclass config loader.
No omegaconf dependency; uses only pyyaml + dataclasses.
"""

from __future__ import annotations
import yaml
import dataclasses
from dataclasses import dataclass, field


# ──────────────────────────────────────────────
# Shared sub-configs
# ──────────────────────────────────────────────

@dataclass
class DataConfig:
    root: str = "./data/stl10"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    unlabeled_split: str = "unlabeled"
    labeled_train_split: str = "train"
    labeled_test_split: str = "test"
    image_size: int = 96


@dataclass
class OptimizerConfig:
    name: str = "adamw"        # "adam" | "adamw" | "sgd"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9      # used only for SGD


@dataclass
class SchedulerConfig:
    warmup_epochs: int = 10
    total_epochs: int = 100
    min_lr: float = 1e-6


@dataclass
class LogConfig:
    log_dir: str = "./runs"
    log_interval: int = 50     # steps between scalar writes
    ckpt_dir: str = "./checkpoints"
    ckpt_interval: int = 10    # epochs between checkpoints


# ──────────────────────────────────────────────
# EBM config
# ──────────────────────────────────────────────

@dataclass
class EBMConfig:
    n_channels: list = field(default_factory=lambda: [3, 64, 128, 256, 512])
    feature_dim: int = 128
    # SGLD
    sgld_steps: int = 20
    sgld_step_size: float = 10.0
    sgld_noise_std: float = 0.005
    replay_buffer_size: int = 10000
    replay_prob: float = 0.95
    # Training
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 42
    epochs: int = 100


# ──────────────────────────────────────────────
# JEPA shared sub-configs
# ──────────────────────────────────────────────

@dataclass
class ViTConfig:
    image_size: int = 96
    patch_size: int = 8
    in_channels: int = 3
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0


@dataclass
class PredictorConfig:
    embed_dim: int = 384
    predictor_embed_dim: int = 192
    depth: int = 4
    num_heads: int = 4


@dataclass
class MaskingConfig:
    # Multi-block masking (I-JEPA §3.2)
    num_target_blocks: int = 4
    target_scale_range: list = field(default_factory=lambda: [0.15, 0.2])
    target_aspect_ratio_range: list = field(default_factory=lambda: [0.75, 1.5])
    context_scale_range: list = field(default_factory=lambda: [0.85, 1.0])
    context_aspect_ratio: float = 1.0
    allow_overlap: bool = False


# ──────────────────────────────────────────────
# I-JEPA config
# ──────────────────────────────────────────────

@dataclass
class IJEPAConfig:
    vit: ViTConfig = field(default_factory=ViTConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    ema_momentum: float = 0.996
    ema_momentum_final: float = 1.0
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1.5e-4))
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 42
    epochs: int = 300


# ──────────────────────────────────────────────
# H-JEPA config
# ──────────────────────────────────────────────

@dataclass
class HJEPAConfig:
    # Each entry: patch_size, embed_dim, depth, num_heads
    levels: list = field(default_factory=lambda: [
        {"patch_size": 8,  "embed_dim": 256, "depth": 4, "num_heads": 4},
        {"patch_size": 16, "embed_dim": 384, "depth": 6, "num_heads": 6},
        {"patch_size": 32, "embed_dim": 512, "depth": 4, "num_heads": 8},
    ])
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    ema_momentum: float = 0.996
    ema_momentum_final: float = 1.0
    cross_level_loss_weight: float = 0.5
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=1.5e-4))
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    log: LogConfig = field(default_factory=LogConfig)
    seed: int = 42
    epochs: int = 300


# ──────────────────────────────────────────────
# Generic YAML loader
# ──────────────────────────────────────────────

_REGISTRY = {
    "ebm": EBMConfig,
    "ijepa": IJEPAConfig,
    "hjepa": HJEPAConfig,
}


def _update_dataclass_from_dict(obj, d: dict) -> None:
    """Recursively update a dataclass instance from a flat or nested dict."""
    for key, value in d.items():
        if not hasattr(obj, key):
            continue
        field_val = getattr(obj, key)
        if dataclasses.is_dataclass(field_val) and isinstance(value, dict):
            _update_dataclass_from_dict(field_val, value)
        else:
            setattr(obj, key, value)


def load_config(yaml_path: str, model: str):
    """
    Load a YAML config file and return a typed dataclass config object.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML config file, e.g. "configs/ebm.yaml"
    model : str
        One of "ebm", "ijepa", "hjepa"

    Returns
    -------
    EBMConfig | IJEPAConfig | HJEPAConfig

    Example
    -------
    >>> cfg = load_config("configs/ijepa.yaml", "ijepa")
    >>> cfg.vit.embed_dim
    384
    """
    if model not in _REGISTRY:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(_REGISTRY)}")

    with open(yaml_path, "r") as f:
        raw: dict = yaml.safe_load(f) or {}

    cfg = _REGISTRY[model]()
    _update_dataclass_from_dict(cfg, raw)
    return cfg
