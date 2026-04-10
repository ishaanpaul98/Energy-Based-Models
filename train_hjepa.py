"""
train_hjepa.py  —  STUB
Entry-point script for H-JEPA training on STL-10.

What's wired up for you (boilerplate):
  - Argument parsing, config loading, seed
  - TensorBoard writer
  - Data loader
  - Model instantiation (HJEPA)
  - One masker per hierarchical level (different patch grids)
  - Optimizer + LR scheduler
  - Checkpoint save/load

What you implement:
  - The training loop body
  - Calling each masker to get context/target indices at each level
  - Forward pass through HJEPA.forward(x, context_ids_per_level,
                                          target_ids_per_level)
  - Computing combined within + cross level loss
  - EMA update for ALL target encoders after each optimizer step
  - Logging

Loss formulation:
-----------------
predictions, targets = model(x, context_ids_per_level, target_ids_per_level)

# Within-level loss: same as I-JEPA per level
L_within = sum(
    jepa_prediction_loss(predictions["within"][l], targets["within"][l])
    for l in range(num_levels)
)

# Cross-level loss: coarser predicts finer
L_cross = sum(
    jepa_prediction_loss(predictions["cross"][l], targets["cross"][l])
    for l in range(num_levels - 1)
)

loss = L_within + cfg.cross_level_loss_weight * L_cross

EMA update (all levels):
------------------------
for l in range(num_levels):
    update_ema(model.context_encs[l], model.target_encs[l], momentum)
"""

import argparse
import os
import random
import numpy as np
import torch

from utils.config import load_config
from utils.logging_utils import make_writer, log_scalars
from data import unlabeled_loader
from models.hjepa import HJEPA
from training.masking import MultiBlockMasking
from training.losses import jepa_prediction_loss
from training.ema import update_ema, cosine_ema_schedule
from training.lr_scheduler import cosine_warmup_scheduler


def parse_args():
    p = argparse.ArgumentParser(description="Train H-JEPA on STL-10")
    p.add_argument("--config",   default="configs/hjepa.yaml")
    p.add_argument("--run-name", default="run1")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",   default=None)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0)


def main():
    args   = parse_args()
    cfg    = load_config(args.config, "hjepa")
    device = torch.device(args.device)

    set_seed(cfg.seed)

    # ── Data ────────────────────────────────────────────────────────────────
    loader = unlabeled_loader(cfg)

    # ── Model ───────────────────────────────────────────────────────────────
    model = HJEPA(cfg).to(device)

    # ── Maskers (one per level, different patch grid sizes) ──────────────────
    # e.g. level 0: 96/8=12, level 1: 96/16=6, level 2: 96/32=3
    maskers = [
        MultiBlockMasking(
            cfg.data.image_size // level["patch_size"],
            cfg.data.image_size // level["patch_size"],
            cfg.masking,
        )
        for level in cfg.levels
    ]

    # ── Optimiser (all context encoders + within/cross predictors) ───────────
    trainable_params = []
    for ctx_enc in model.context_encs:
        trainable_params += list(ctx_enc.parameters())
    for pred in model.within_preds:
        trainable_params += list(pred.parameters())
    trainable_params += list(model.cross_pred.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        total_epochs=cfg.epochs,
        min_lr_ratio=cfg.scheduler.min_lr / cfg.optimizer.lr,
    )

    # ── TensorBoard ──────────────────────────────────────────────────────────
    writer = make_writer(cfg.log.log_dir, args.run_name)

    # ── AMP (Automatic Mixed Precision) ──────────────────────────────────────
    use_amp   = device.type == "cuda"
    amp_dtype = torch.bfloat16   # BF16 is natively fast on RTX 5000-series (Blackwell)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Optional resume ──────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    num_levels = len(cfg.levels)

    # ── Training loop ────────────────────────────────────────────────────────
    # TODO: implement training loop here.
    #
    # Wrap your forward pass + loss in an autocast block:
    #
    #   with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
    #       predictions, targets = model(x, context_ids_per_level, target_ids_per_level)
    #       L_within = sum(jepa_prediction_loss(predictions["within"][l],
    #                                           targets["within"][l])
    #                      for l in range(num_levels))
    #       L_cross  = sum(jepa_prediction_loss(predictions["cross"][l],
    #                                           targets["cross"][l])
    #                      for l in range(num_levels - 1))
    #       loss = L_within + cfg.cross_level_loss_weight * L_cross
    #
    # Use the scaler for backward:
    #
    #   optimizer.zero_grad()
    #   scaler.scale(loss).backward()
    #   scaler.step(optimizer)
    #   scaler.update()
    #
    # Other hints:
    #   - For each batch: maskers[l](B) -> context_ids_per_level[l], target_ids_per_level[l]
    #   - After scaler.update(), call update_ema for each level:
    #       for l in range(num_levels):
    #           update_ema(model.context_encs[l], model.target_encs[l], m)
    #   - Log L_within, L_cross, total loss each cfg.log.log_interval steps

    raise NotImplementedError("Implement the training loop in train_hjepa.py")

    writer.close()


if __name__ == "__main__":
    main()
