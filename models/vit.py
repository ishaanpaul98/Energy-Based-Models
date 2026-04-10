"""
models/vit.py  —  STUB
Vision Transformer (ViT) backbone shared by I-JEPA and H-JEPA.

References
----------
Dosovitskiy et al. (2020). "An Image is Worth 16x16 Words: Transformers for
    Image Recognition at Scale."  ICLR 2021.  https://arxiv.org/abs/2010.11929

Assran et al. (2023). I-JEPA uses a ViT-S/8 for STL-10 scale experiments.
    https://arxiv.org/abs/2301.08243

Design notes
------------
PatchEmbed
    The most efficient patch embedding is a single Conv2d with kernel_size=patch_size
    and stride=patch_size. This is equivalent to independently projecting each patch.

Positional embeddings
    Use fixed 2D sinusoidal embeddings OR learnable nn.Parameter of shape
    [1, num_patches, embed_dim]. Both work. Learnable is simpler to implement.

[CLS] token
    OPTIONAL for I-JEPA. The paper uses mean-pooling of patch tokens for the
    linear probe, not a CLS token. You may add one if you prefer, but it is
    not required.

Context vs. target encoder usage
    - Context encoder: receives ONLY unmasked patch tokens.
      Pass keep_ids=[B, N_keep] to forward() to index-select those patches.
    - Target encoder: receives ALL N patches.
      Pass keep_ids=None (default) to forward().

    IMPORTANT: when index-selecting tokens, you must also select the matching
    positional embeddings so each token still carries its absolute position.
"""

import torch
import torch.nn as nn
from utils.config import ViTConfig


class PatchEmbed(nn.Module):
    """
    Convert image [B, C, H, W] -> patch token sequence [B, N, embed_dim]
    where N = (H // patch_size) * (W // patch_size).

    Parameters
    ----------
    cfg : ViTConfig
    """

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()

        # TODO: define self.proj = nn.Conv2d(
        #     cfg.in_channels, cfg.embed_dim,
        #     kernel_size=cfg.patch_size, stride=cfg.patch_size
        # )
        # TODO: compute and store self.num_patches as an int:
        #   num_patches = (cfg.image_size // cfg.patch_size) ** 2

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, C, H, W]

        Returns
        -------
        tokens : [B, N, embed_dim]

        Hint:
            out = self.proj(x)           # [B, embed_dim, H/p, W/p]
            out = out.flatten(2)         # [B, embed_dim, N]
            out = out.transpose(1, 2)    # [B, N, embed_dim]
        """
        raise NotImplementedError


class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block:
        y = x + Attention(LayerNorm(x))
        z = y + MLP(LayerNorm(y))

    Parameters
    ----------
    embed_dim    : int
    num_heads    : int
    mlp_ratio    : float — hidden MLP dim = embed_dim * mlp_ratio
    dropout      : float — applied after attention output projection and MLP
    attn_dropout : float — applied inside attention (nn.MultiheadAttention dropout arg)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # TODO: define:
        #   self.norm1 = nn.LayerNorm(embed_dim)
        #   self.attn  = nn.MultiheadAttention(
        #       embed_dim, num_heads, dropout=attn_dropout, batch_first=True
        #   )
        #   self.norm2 = nn.LayerNorm(embed_dim)
        #   mlp_hidden = int(embed_dim * mlp_ratio)
        #   self.mlp   = nn.Sequential(
        #       nn.Linear(embed_dim, mlp_hidden),
        #       nn.GELU(),
        #       nn.Dropout(dropout),
        #       nn.Linear(mlp_hidden, embed_dim),
        #       nn.Dropout(dropout),
        #   )
        #   self.drop  = nn.Dropout(dropout)  # for attention output

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, N, embed_dim]

        Returns
        -------
        x : [B, N, embed_dim]

        Hint (pre-norm style):
            normed = self.norm1(x)
            attn_out, _ = self.attn(normed, normed, normed)
            x = x + self.drop(attn_out)
            x = x + self.mlp(self.norm2(x))
            return x
        """
        raise NotImplementedError


class VisionTransformer(nn.Module):
    """
    Full ViT encoder.

    This is the backbone used by both I-JEPA and H-JEPA.
    One instance serves as the context encoder (online, trained with gradients),
    another as the target encoder (EMA copy, no gradients ever stored).

    Parameters
    ----------
    cfg : ViTConfig
    """

    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()

        # TODO: define:
        #   self.patch_embed = PatchEmbed(cfg)
        #   self.pos_embed   = nn.Parameter(
        #       torch.zeros(1, self.patch_embed.num_patches, cfg.embed_dim)
        #   )
        #   self.blocks = nn.ModuleList([
        #       TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio,
        #                        cfg.dropout, cfg.attn_dropout)
        #       for _ in range(cfg.depth)
        #   ])
        #   self.norm = nn.LayerNorm(cfg.embed_dim)
        #
        # Also initialize pos_embed (e.g. nn.init.trunc_normal_(self.pos_embed, std=0.02))

        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        keep_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x        : [B, C, H, W]   input images
        keep_ids : [B, N_keep]    LongTensor of patch indices to KEEP.
                   - None  (default) → target encoder mode: process all N patches.
                   - Provided       → context encoder mode: keep only these patches.

        Returns
        -------
        tokens : [B, N_out, embed_dim]
            N_out = N when keep_ids is None, else N_keep.

        Implementation steps:
        1. tokens = self.patch_embed(x)                  # [B, N, D]
        2. tokens = tokens + self.pos_embed              # broadcast add positional emb
        3. if keep_ids is not None:
               # Index-select the kept tokens (AND their positional embeddings
               # are already baked in from step 2, so no separate PE lookup needed)
               tokens = tokens[torch.arange(B).unsqueeze(1), keep_ids]
                        # shape: [B, N_keep, D]
        4. for block in self.blocks:
               tokens = block(tokens)
        5. return self.norm(tokens)
        """
        raise NotImplementedError
