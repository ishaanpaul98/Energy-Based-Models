"""
models/hjepa.py  —  STUB
H-JEPA: Hierarchical Joint-Embedding Predictive Architecture.

Extends I-JEPA to multiple levels of spatial abstraction.  Three ViT encoders
operate at different patch resolutions, giving representations at different
granularities.  Predictions are made both WITHIN each level (like I-JEPA) and
ACROSS levels (coarser level predicts finer-level representations).

Architecture overview
---------------------
image
 ├─► Level 0 context_enc  (patch 8,  144 tokens) ──► z0_ctx
 ├─► Level 1 context_enc  (patch 16,  36 tokens) ──► z1_ctx
 └─► Level 2 context_enc  (patch 32,   9 tokens) ──► z2_ctx

 ├─► Level 0 target_enc   (EMA, patch 8)  ──► z0_tgt   (within-level targets)
 ├─► Level 1 target_enc   (EMA, patch 16) ──► z1_tgt
 └─► Level 2 target_enc   (EMA, patch 32) ──► z2_tgt

Predictors:
  within_predictors[i] : z_i_ctx → predicted z_i_tgt  (same level, like I-JEPA)
  cross_predictor      : z_{i+1}_ctx → predicted z_i_tgt  (coarser predicts finer)

Loss:
  L_within = sum_i  jepa_prediction_loss(within_pred_i, z_i_tgt)
  L_cross  = sum_i  jepa_prediction_loss(cross_pred_i,  z_{i-1}_tgt)
  L_total  = L_within + cfg.cross_level_loss_weight * L_cross

Design notes
------------
- Each level is an independent VisionTransformer with its own patch size and
  embed_dim.  There is NO weight sharing between levels by default.
- The cross-level predictor maps from level i+1's embed_dim to level i's
  embed_dim.  You will need a projection if dimensions differ.
- Keep three independent target encoders (one per level).  Each is an EMA copy
  of its corresponding context encoder.
- Masking is applied independently at each resolution level.  The masking
  indices from level 0 can be downsampled to level 1/2 (divide patch indices
  by the ratio of patch sizes) or re-sampled independently.

References
----------
LeCun (2022). "A Path Towards Autonomous Machine Intelligence."
    https://openreview.net/forum?id=BZ5a1r-kVsf  (H-JEPA concept)

Assran et al. (2023). I-JEPA (foundation for H-JEPA).
    https://arxiv.org/abs/2301.08243
"""

import torch
import torch.nn as nn
from utils.config import HJEPAConfig, ViTConfig
from models.vit import VisionTransformer
from models.ijepa import Predictor


class HJEPA(nn.Module):
    """
    Hierarchical JEPA with 3 levels.

    Parameters
    ----------
    cfg : HJEPAConfig
        cfg.levels : list of dicts, each with keys:
            patch_size, embed_dim, depth, num_heads

    Attributes
    ----------
    context_encs   : nn.ModuleList of VisionTransformer  (one per level, online)
    target_encs    : nn.ModuleList of VisionTransformer  (one per level, EMA)
    within_preds   : nn.ModuleList of Predictor          (within-level, one per level)
    cross_pred     : nn.Module                           (cross-level predictor)
    """

    def __init__(self, cfg: HJEPAConfig) -> None:
        super().__init__()

        # TODO: For each level in cfg.levels, build a ViTConfig and instantiate
        #       a context encoder and a target encoder (EMA copy).
        #
        #   self.context_encs = nn.ModuleList()
        #   self.target_encs  = nn.ModuleList()
        #   for level_cfg in cfg.levels:
        #       vit_cfg = ViTConfig(
        #           image_size=cfg.data.image_size,   # NOTE: access from cfg.data
        #           patch_size=level_cfg["patch_size"],
        #           embed_dim=level_cfg["embed_dim"],
        #           depth=level_cfg["depth"],
        #           num_heads=level_cfg["num_heads"],
        #       )
        #       ctx = VisionTransformer(vit_cfg)
        #       tgt = VisionTransformer(vit_cfg)
        #       tgt.load_state_dict(ctx.state_dict())
        #       for p in tgt.parameters():
        #           p.requires_grad = False
        #       self.context_encs.append(ctx)
        #       self.target_encs.append(tgt)
        #
        # TODO: Build within_preds: one Predictor per level.
        #   self.within_preds = nn.ModuleList()
        #   for level_cfg in cfg.levels:
        #       ... (you will need a lightweight config object or pass dims directly)
        #
        # TODO: Build cross_pred for cross-level prediction.
        #   The simplest design: one predictor that maps level[i+1]'s context tokens
        #   to predict level[i]'s target tokens.  Since embed_dims differ across
        #   levels, you may need a projection layer before/after the predictor.
        #   e.g. self.cross_proj = nn.ModuleList([
        #       nn.Linear(cfg.levels[i+1]["embed_dim"], cfg.levels[i]["embed_dim"])
        #       for i in range(len(cfg.levels) - 1)
        #   ])

        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        context_ids_per_level: list[torch.Tensor],
        target_ids_per_level: list[list[torch.Tensor]],
    ) -> tuple[dict, dict]:
        """
        Parameters
        ----------
        x : [B, C, H, W]  full images

        context_ids_per_level : list of L tensors, each [B, N_ctx_l]
            Unmasked patch indices at each level l.
            N_ctx_l differs across levels because patch counts differ.

        target_ids_per_level : list of L lists, each containing M tensors [B, N_tgt_m]
            Target block indices at each level l, for each of M blocks.

        Returns
        -------
        predictions : dict with keys "within" and "cross"
            "within" : list of L  lists of M tensors [B, N_tgt_m, D_l]
            "cross"  : list of (L-1) lists of M tensors [B, N_tgt_m, D_{l-1}]
                       (coarser level l predicts finer level l-1)

        targets : dict with keys "within" and "cross"
            Same structure as predictions; these are the target encoder outputs.

        Implementation outline:
        -----------------------
        B = x.shape[0]
        within_preds, within_tgts = [], []
        cross_preds,  cross_tgts  = [], []

        # --- Within-level predictions (I-JEPA style per level) ---
        ctx_tokens = []
        for l, (ctx_enc, tgt_enc, predictor) in enumerate(
            zip(self.context_encs, self.target_encs, self.within_preds)
        ):
            z_ctx = ctx_enc(x, keep_ids=context_ids_per_level[l])
            ctx_tokens.append(z_ctx)

            with torch.no_grad():
                z_all = tgt_enc(x)
                tgts_l = [z_all[torch.arange(B).unsqueeze(1), ids]
                           for ids in target_ids_per_level[l]]
            within_tgts.append(tgts_l)

            preds_l = predictor(z_ctx, context_ids_per_level[l], target_ids_per_level[l])
            within_preds.append(preds_l)

        # --- Cross-level predictions (level l+1 context -> level l targets) ---
        for l in range(len(self.context_encs) - 1):
            z_coarse = ctx_tokens[l + 1]          # coarser context tokens
            z_coarse_proj = self.cross_proj[l](z_coarse)  # project to finer dim
            # Use within_tgts[l] as the cross-level supervision target
            # (coarser encoder predicts the finer-level representations)
            preds_cross = self.cross_pred(
                z_coarse_proj,
                context_ids_per_level[l + 1],     # NOTE: ids are in coarser grid
                target_ids_per_level[l],           # predict finer-level positions
            )
            cross_preds.append(preds_cross)
            cross_tgts.append(within_tgts[l])     # same targets as within-level l

        predictions = {"within": within_preds, "cross": cross_preds}
        targets     = {"within": within_tgts,  "cross": cross_tgts}
        return predictions, targets
        """
        raise NotImplementedError
