"""
models/ijepa.py  —  STUB
I-JEPA: Image Joint-Embedding Predictive Architecture.

READ THIS PAPER FIRST before implementing:
    Assran et al. (2023). "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture."  CVPR 2023.
    https://arxiv.org/abs/2301.08243

Architecture overview
---------------------
                        ┌──────────────┐
    image (full)   ───► │ target_enc   │ ──► z_target  [B, N, D]   (no grad)
                        │ (EMA of ctx) │
                        └──────────────┘

                        ┌──────────────┐
    image (ctx     ───► │ context_enc  │ ──► z_ctx     [B, N_ctx, D]
    patches only)       │ (online ViT) │
                        └──────────────┘
                                │
                                ▼
                        ┌──────────────┐
    target positions ──►│  predictor   │ ──► z_pred_m  [B, N_tgt, D]  per block m
                        │ (narrow ViT) │
                        └──────────────┘

Training loss (eq. 1 in paper):
    L = (1/M) * sum_m || z_pred_m - sg(z_target_m) ||_2^2
    where sg = stop-gradient (target encoder receives no gradients ever).
    The loss is computed per patch token in each target block, then averaged.

Key design decisions
--------------------
- The predictor takes context tokens + MASK TOKENS at each target block position.
  Mask tokens are a learnable vector; they are given the positional embedding of
  the target position so the predictor knows WHERE it should predict.
- A single context encoder forward pass covers all M target blocks (batch them).
- Target representations are L2-normalised before computing the loss (optional
  in the original code, but improves stability).
- The target encoder is NEVER updated by gradients. Only EMA from context_enc.
  Use `with torch.no_grad():` for the target encoder forward pass.
  Use `training/ema.py:update_ema()` after every optimizer step.
"""

import torch
import torch.nn as nn
from utils.config import IJEPAConfig
from models.vit import VisionTransformer, TransformerBlock


class Predictor(nn.Module):
    """
    Narrow ViT that predicts target patch representations from context tokens.

    The predictor is deliberately shallower and narrower than the encoder to
    prevent it from learning a trivial identity mapping.

    Parameters
    ----------
    cfg : IJEPAConfig
        Uses cfg.predictor (PredictorConfig) and cfg.vit.embed_dim.

    Architecture (paper §3.3)
    -------------------------
    1. input_proj  : Linear(embed_dim -> predictor_embed_dim)
       Project context tokens down to the narrower predictor width.

    2. mask_token  : nn.Parameter([1, 1, predictor_embed_dim])
       Learnable vector broadcast-added to positional embeddings at target positions.
       Represents "I need to predict the patch at this position."

    3. pos_embed   : nn.Parameter([1, num_patches, predictor_embed_dim])
       Full-resolution positional embeddings for all N patches. These are indexed
       by both context_ids and each set of target_ids to give every token
       its absolute position before running the transformer.

    4. blocks      : nn.ModuleList of TransformerBlock
       cfg.predictor.depth narrow transformer blocks.

    5. norm        : nn.LayerNorm(predictor_embed_dim)

    6. output_proj : Linear(predictor_embed_dim -> embed_dim)
       Project predictions back up to match the target encoder output dimension.
    """

    def __init__(self, cfg: IJEPAConfig) -> None:
        super().__init__()
        D  = cfg.vit.embed_dim
        Dp = cfg.predictor.predictor_embed_dim
        N  = (cfg.vit.image_size // cfg.vit.patch_size) ** 2

        # TODO: define:
        #   self.input_proj  = nn.Linear(D, Dp)
        #   self.mask_token  = nn.Parameter(torch.zeros(1, 1, Dp))
        #   self.pos_embed   = nn.Parameter(torch.zeros(1, N, Dp))
        #   self.blocks      = nn.ModuleList([
        #       TransformerBlock(Dp, cfg.predictor.num_heads)
        #       for _ in range(cfg.predictor.depth)
        #   ])
        #   self.norm        = nn.LayerNorm(Dp)
        #   self.output_proj = nn.Linear(Dp, D)
        #
        # Initialise parameters:
        #   nn.init.trunc_normal_(self.mask_token, std=0.02)
        #   nn.init.trunc_normal_(self.pos_embed,  std=0.02)

        raise NotImplementedError

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_ids: torch.Tensor,
        target_ids_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        context_tokens  : [B, N_ctx, D]
            Output of the context encoder for the unmasked patches.
        context_ids     : [B, N_ctx]  LongTensor
            Patch indices of the context tokens. Used to look up their
            positional embeddings so the predictor knows WHERE each context
            token came from.
        target_ids_list : list of M tensors, each [B, N_tgt_m]  LongTensor
            Patch indices of each target block m.

        Returns
        -------
        predictions : list of M tensors, each [B, N_tgt_m, D]
            Predicted representations for each target block.
            D matches the target encoder output dimension (cfg.vit.embed_dim).

        Implementation sketch (process all M blocks in one batched forward pass):
        -------------------------------------------------------------------------
        B = context_tokens.shape[0]

        1. Project context tokens to predictor width:
               ctx = self.input_proj(context_tokens)   # [B, N_ctx, Dp]

        2. Add positional embeddings to context tokens:
               ctx_pe = self.pos_embed.expand(B, -1, -1)  # [B, N, Dp]
               ctx = ctx + ctx_pe[torch.arange(B).unsqueeze(1), context_ids]

        3. For each target block m:
               # Create mask tokens for this block's positions
               tgt_ids = target_ids_list[m]           # [B, N_tgt]
               mask_toks = self.mask_token.expand(B, tgt_ids.shape[1], -1).clone()
               # Add positional embeddings so predictor knows the target location
               mask_toks = mask_toks + ctx_pe[torch.arange(B).unsqueeze(1), tgt_ids]

               # Concatenate: [context tokens | mask tokens]
               tokens = torch.cat([ctx, mask_toks], dim=1)   # [B, N_ctx+N_tgt, Dp]

               # Run transformer blocks
               for block in self.blocks:
                   tokens = block(tokens)
               tokens = self.norm(tokens)

               # Extract only the mask token positions (last N_tgt tokens)
               pred_m = tokens[:, -tgt_ids.shape[1]:]        # [B, N_tgt, Dp]
               pred_m = self.output_proj(pred_m)              # [B, N_tgt, D]
               predictions.append(pred_m)

        return predictions
        """
        raise NotImplementedError


class IJEPA(nn.Module):
    """
    Full I-JEPA model: context encoder + target encoder + predictor.

    The target encoder is initialised as a copy of the context encoder and
    is subsequently updated ONLY via EMA (never via backprop).

    Parameters
    ----------
    cfg : IJEPAConfig

    Attributes
    ----------
    context_enc : VisionTransformer  (trained online)
    target_enc  : VisionTransformer  (EMA copy, no grad)
    predictor   : Predictor
    """

    def __init__(self, cfg: IJEPAConfig) -> None:
        super().__init__()

        # TODO: instantiate context_enc, target_enc, predictor
        #   self.context_enc = VisionTransformer(cfg.vit)
        #   self.target_enc  = VisionTransformer(cfg.vit)
        #   self.predictor   = Predictor(cfg)
        #
        # Copy context_enc weights into target_enc and freeze target_enc:
        #   self.target_enc.load_state_dict(self.context_enc.state_dict())
        #   for p in self.target_enc.parameters():
        #       p.requires_grad = False

        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        context_ids: torch.Tensor,
        target_ids_list: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Single forward pass for one training step.

        Parameters
        ----------
        x               : [B, C, H, W]  full images (normalized)
        context_ids     : [B, N_ctx]    unmasked patch indices for context encoder
        target_ids_list : list of M tensors [B, N_tgt_m]  target block indices

        Returns
        -------
        predictions  : list of M tensors [B, N_tgt_m, D]  predictor outputs
        targets      : list of M tensors [B, N_tgt_m, D]  target encoder outputs
                       (detached — no gradients flow through target encoder)

        Implementation:
        ---------------
        # 1. Context encoder (online, gradients flow)
        z_ctx = self.context_enc(x, keep_ids=context_ids)  # [B, N_ctx, D]

        # 2. Target encoder (EMA copy, no gradients)
        with torch.no_grad():
            z_all = self.target_enc(x)                     # [B, N, D]
            # Extract target representations for each block
            targets = [
                z_all[torch.arange(B).unsqueeze(1), tgt_ids]
                for tgt_ids in target_ids_list
            ]                                              # list of [B, N_tgt_m, D]

        # 3. Predictor
        predictions = self.predictor(z_ctx, context_ids, target_ids_list)

        return predictions, targets
        """
        raise NotImplementedError
