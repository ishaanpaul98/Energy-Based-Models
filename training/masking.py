"""
training/masking.py  —  STUB
Multi-block masking strategy for I-JEPA and H-JEPA.

READ §3.2 of the I-JEPA paper before implementing:
    Assran et al. (2023). https://arxiv.org/abs/2301.08243

Goal
----
Given an image divided into an H_p x W_p grid of patches, produce:
  - context_ids   : [B, N_ctx]   indices of patches the context encoder sees
  - target_ids    : list of M tensors, each [B, N_tgt_m]  target block indices

The context is a LARGE crop of the patch grid (high coverage, ~85% of patches).
The targets are M small rectangular blocks (~15-20% of patches each) sampled
with random scale and aspect ratio.

Algorithm (per image in the batch)
-----------------------------------
1. Sample M target blocks:
   For each block m = 1..M:
     a. Sample scale s ~ Uniform(target_scale_range)
        area_tgt = s * H_p * W_p
     b. Sample log aspect ratio r ~ Uniform(log(min_ar), log(max_ar))
        ar = exp(r)
        h_m = round(sqrt(area_tgt / ar))   clamped to [1, H_p]
        w_m = round(sqrt(area_tgt * ar))   clamped to [1, W_p]
     c. Sample top-left corner: i ~ [0, H_p - h_m], j ~ [0, W_p - w_m]
     d. Collect all patch indices in the h_m x w_m rectangle.

2. Sample 1 context block:
   a. Sample scale s_ctx ~ Uniform(context_scale_range)
      area_ctx = s_ctx * H_p * W_p
   b. Use context_aspect_ratio (default 1.0) to get h_ctx, w_ctx.
   c. Sample top-left corner.
   d. Context indices = the sampled rectangle MINUS the union of all target blocks.
      (The context encoder CANNOT see any target patch.)

3. Return context_ids, target_ids_list.

Notes
-----
- For the batch dimension, run the algorithm independently per image.
  The resulting context/target sets may have DIFFERENT SIZES per image.
  Pad with -1 or use a ragged representation, OR enforce same size via
  rejection sampling / fixed block sizes.  The simplest approach for a first
  implementation: sample fixed h_m, w_m that give the same count every image.

- Patch indices use ROW-MAJOR order: index = row * W_p + col.

- If allow_overlap=False, re-sample target blocks that overlap existing ones.
  For a first pass, overlapping is fine (allow_overlap=True behavior).
"""

from __future__ import annotations
import math
import random
import torch
from utils.config import MaskingConfig


class MultiBlockMasking:
    """
    Generates context / target patch index sets for a batch of images.

    Parameters
    ----------
    num_patches_h : int  number of patch rows    (image_size // patch_size)
    num_patches_w : int  number of patch columns (image_size // patch_size)
    cfg           : MaskingConfig

    Usage
    -----
    masker = MultiBlockMasking(12, 12, cfg.masking)
    context_ids, target_ids_list = masker(batch_size=B)
    # context_ids      : torch.LongTensor [B, N_ctx]
    # target_ids_list  : list of M torch.LongTensor [B, N_tgt_m]
    """

    def __init__(
        self,
        num_patches_h: int,
        num_patches_w: int,
        cfg: MaskingConfig,
    ) -> None:
        self.H = num_patches_h
        self.W = num_patches_w
        self.cfg = cfg

    def __call__(self, batch_size: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Parameters
        ----------
        batch_size : int

        Returns
        -------
        context_ids     : torch.LongTensor [B, N_ctx]
        target_ids_list : list of M torch.LongTensor [B, N_tgt_m]

        Implementation hint — single-image helper:
        ------------------------------------------
        def _sample_block(H, W, scale_range, ar_range):
            s  = random.uniform(*scale_range)
            ar = math.exp(random.uniform(math.log(ar_range[0]),
                                          math.log(ar_range[1])))
            area = s * H * W
            h = max(1, min(H, round(math.sqrt(area / ar))))
            w = max(1, min(W, round(math.sqrt(area * ar))))
            top  = random.randint(0, H - h)
            left = random.randint(0, W - w)
            ids = set()
            for r in range(top, top + h):
                for c in range(left, left + w):
                    ids.add(r * W + c)
            return ids

        For each image in the batch:
          1. Sample M target blocks with target_scale_range, target_aspect_ratio_range
          2. context_set = _sample_block(context_scale_range, context_aspect_ratio=1.0)
             context_set -= union of all target blocks
          3. Convert sets to sorted LongTensors

        Batch by padding to max length or (simpler) ensure same size per block.
        """
        raise NotImplementedError

    def _sample_block_ids(
        self,
        scale_range: list,
        ar_range: list,
    ) -> set:
        """
        Sample a single rectangular block of patch indices.

        Parameters
        ----------
        scale_range : [min_scale, max_scale] — block area as fraction of grid
        ar_range    : [min_ar, max_ar]       — aspect ratio range

        Returns
        -------
        set of int  : patch indices (row-major) inside the sampled rectangle
        """
        # TODO: implement the block sampling logic described above
        raise NotImplementedError
