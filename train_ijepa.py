"""
train_ijepa.py  —  STUB
Entry-point script for I-JEPA training on STL-10.

What's wired up for you (boilerplate):
  - Argument parsing (--config, --run-name, --device)
  - Config loading + seed
  - TensorBoard writer
  - Data loader
  - Model instantiation (IJEPA — context + target encoder + predictor)
  - Masker instantiation (MultiBlockMasking)
  - Optimizer + LR scheduler
  - Checkpoint save/load

What you implement:
  - The training loop body
  - Calling the masker to get context_ids and target_ids_list each step
  - Forward pass through IJEPA.forward(x, context_ids, target_ids_list)
  - Computing jepa_prediction_loss(predictions, targets)
  - EMA update of target encoder after each optimizer step
  - Logging loss and EMA momentum

Suggested loop structure:
-------------------------
total_steps = cfg.epochs * len(loader)

for epoch in range(start_epoch, cfg.epochs):
    for step, (x, _) in enumerate(loader):
        x = x.to(device)
        B = x.shape[0]

        # 1. Sample masks
        context_ids, target_ids_list = masker(B)
        context_ids   = context_ids.to(device)
        target_ids_list = [t.to(device) for t in target_ids_list]

        # 2. Forward pass
        predictions, targets = model(x, context_ids, target_ids_list)

        # 3. Loss
        loss = jepa_prediction_loss(predictions, targets, normalize_targets=True)

        # 4. Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. EMA update of target encoder
        global_step = epoch * len(loader) + step
        m = cosine_ema_schedule(cfg.ema_momentum, cfg.ema_momentum_final,
                                global_step, total_steps)
        update_ema(model.context_enc, model.target_enc, m)

        # 6. Logging
        if global_step % cfg.log.log_interval == 0:
            log_scalars(writer, {"loss/jepa": loss.item(), "ema/momentum": m},
                        global_step)

    scheduler.step()
"""

import argparse
import os
import random
import numpy as np
import torch

from utils.config import load_config
from utils.logging_utils import make_writer, log_scalars
from data import unlabeled_loader
from models.ijepa import IJEPA
from training.masking import MultiBlockMasking
from training.losses import jepa_prediction_loss
from training.ema import update_ema, cosine_ema_schedule
from training.lr_scheduler import cosine_warmup_scheduler


def parse_args():
    p = argparse.ArgumentParser(description="Train I-JEPA on STL-10")
    p.add_argument("--config",   default="configs/ijepa.yaml")
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
    cfg    = load_config(args.config, "ijepa")
    device = torch.device(args.device)

    set_seed(cfg.seed)

    # ── Data ────────────────────────────────────────────────────────────────
    loader = unlabeled_loader(cfg)

    # ── Model ───────────────────────────────────────────────────────────────
    model = IJEPA(cfg).to(device)

    # ── Masker ──────────────────────────────────────────────────────────────
    num_patches = cfg.vit.image_size // cfg.vit.patch_size  # e.g. 12
    masker = MultiBlockMasking(num_patches, num_patches, cfg.masking)

    # ── Optimiser (only context_enc + predictor; target_enc has no grad) ────
    trainable = list(model.context_enc.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
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

    # ── Training loop ────────────────────────────────────────────────────────
    # TODO: implement training loop here.
    #
    # Wrap your forward pass + loss in an autocast block:
    #
    #   with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
    #       predictions, targets = model(x, context_ids, target_ids_list)
    #       loss = jepa_prediction_loss(predictions, targets, normalize_targets=True)
    #
    # Use the scaler for backward:
    #
    #   optimizer.zero_grad()
    #   scaler.scale(loss).backward()
    #   scaler.step(optimizer)
    #   scaler.update()
    #
    # Other hints:
    #   - Call masker(B) each step to get fresh context_ids, target_ids_list
    #   - After scaler.update(), call update_ema(model.context_enc, model.target_enc, m)
    #     where m = cosine_ema_schedule(cfg.ema_momentum, cfg.ema_momentum_final,
    #                                   global_step, total_steps)
    #   - Log loss and EMA momentum each cfg.log.log_interval steps

    raise NotImplementedError("Implement the training loop in train_ijepa.py")

    writer.close()


if __name__ == "__main__":
    main()
