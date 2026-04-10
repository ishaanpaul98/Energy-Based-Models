"""
train_ebm.py  —  STUB
Entry-point script for EBM training on STL-10.

What's wired up for you (boilerplate):
  - Argument parsing (--config, --run-name, --device)
  - Config loading
  - Reproducibility seed
  - TensorBoard writer
  - Data loader
  - Model instantiation
  - Optimizer + LR scheduler
  - Checkpoint directory creation
  - Checkpoint save/load helpers

What you implement:
  - The training loop body (one epoch)
  - SGLD sampling via ReplayBuffer + sgld_sample
  - Contrastive divergence loss
  - Logging scalars and sample images

Suggested loop structure:
-------------------------
for epoch in range(cfg.epochs):
    for step, (x_pos, _) in enumerate(loader):
        x_pos = x_pos.to(device)

        # 1. Sample negatives via SGLD
        x_init, _ = replay_buffer.sample(x_pos.shape[0], cfg.replay_prob, device)
        x_neg = sgld_sample(model, x_init, cfg.sgld_steps,
                            cfg.sgld_step_size, cfg.sgld_noise_std)
        replay_buffer.push(x_neg)

        # 2. Compute energies
        model.train()
        e_pos = model(x_pos)
        e_neg = model(x_neg)

        # 3. Contrastive divergence loss
        loss = contrastive_divergence_loss(e_pos, e_neg, l2_reg_weight=1.0)

        # 4. Backward + step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5. Logging
        global_step = epoch * len(loader) + step
        if global_step % cfg.log.log_interval == 0:
            log_scalars(writer, {
                "loss/cd":       loss.item(),
                "energy/pos":    e_pos.mean().item(),
                "energy/neg":    e_neg.mean().item(),
            }, global_step)

    scheduler.step()
    # Periodically log sample images and save checkpoint
"""

import argparse
import os
import random
import numpy as np
import torch

from utils.config import load_config
from utils.logging_utils import make_writer, log_scalars, log_images
from data import unlabeled_loader
from models.ebm_net import EnergyNet
from training.sgld import ReplayBuffer, sgld_sample
from training.losses import contrastive_divergence_loss
from training.lr_scheduler import cosine_warmup_scheduler


def parse_args():
    p = argparse.ArgumentParser(description="Train EBM on STL-10")
    p.add_argument("--config",   default="configs/ebm.yaml", help="Path to YAML config")
    p.add_argument("--run-name", default="run1",             help="TensorBoard run name")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",   default=None,               help="Path to checkpoint to resume from")
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
    cfg    = load_config(args.config, "ebm")
    device = torch.device(args.device)

    set_seed(cfg.seed)

    # ── Data ────────────────────────────────────────────────────────────────
    loader = unlabeled_loader(cfg)

    # ── Model ───────────────────────────────────────────────────────────────
    model = EnergyNet(cfg).to(device)

    # ── Replay buffer ────────────────────────────────────────────────────────
    img_shape = (3, cfg.data.image_size, cfg.data.image_size)
    replay_buffer = ReplayBuffer(cfg.replay_buffer_size, img_shape)

    # ── Optimiser + scheduler ────────────────────────────────────────────────
    if cfg.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=cfg.optimizer.momentum,
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
    # Uses Tensor Cores on the RTX GPU for ~2-4x throughput on matmuls/convs.
    # BF16 is preferred on Blackwell (sm_120); falls back gracefully on CPU.
    use_amp  = device.type == "cuda"
    amp_dtype = torch.bfloat16   # BF16 is natively fast on RTX 5000-series
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Optional resume ──────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # ── Training loop ────────────────────────────────────────────────────────
    # TODO: implement training loop here.
    #
    # Wrap your forward pass + loss in an autocast block to use AMP:
    #
    #   with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
    #       e_pos = model(x_pos)
    #       e_neg = model(x_neg)
    #       loss  = contrastive_divergence_loss(e_pos, e_neg, l2_reg_weight=1.0)
    #
    # Then use the scaler for the backward pass:
    #
    #   optimizer.zero_grad()
    #   scaler.scale(loss).backward()
    #   scaler.step(optimizer)
    #   scaler.update()
    #
    # NOTE: SGLD sampling should run OUTSIDE the autocast block in FP32
    # because the Langevin gradient update needs full precision.
    #
    # Other hints:
    #   - Use `loader` (yields (x, _) pairs; ignore the label)
    #   - Use `replay_buffer.sample(...)` + `sgld_sample(...)` for negatives
    #   - Use `log_scalars(writer, {...}, step)` for logging
    #   - Use `log_images(writer, "samples/neg", x_neg * 0.5 + 0.5, step)`
    #     to visualise SGLD samples (de-normalise from normalised space if needed)
    #   - Save checkpoint every cfg.log.ckpt_interval epochs:
    #       save_checkpoint(
    #           {"epoch": epoch, "model": model.state_dict(),
    #            "optimizer": optimizer.state_dict(),
    #            "scheduler": scheduler.state_dict()},
    #           os.path.join(cfg.log.ckpt_dir, f"epoch_{epoch:04d}.pt")
    #       )

    raise NotImplementedError("Implement the training loop in train_ebm.py")

    writer.close()


if __name__ == "__main__":
    main()
