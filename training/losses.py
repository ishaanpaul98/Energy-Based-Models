"""
training/losses.py  —  STUB
Loss functions for EBM and JEPA-style training.

EBM loss (Contrastive Divergence)
----------------------------------
The EBM is trained by contrasting energies of real data ("positive" samples)
against SGLD-generated samples ("negative" samples).

The contrastive divergence loss minimises:
    L_CD = E_{x+ ~ data}[E_theta(x+)] - E_{x- ~ p_theta}[E_theta(x-)]

Intuitively: push DOWN the energy of real data, push UP the energy of generated
samples.  The gradient of L_CD w.r.t. theta approximates the gradient of the
negative log-likelihood.

Reference: Hinton (2002). "Training Products of Experts by Minimizing
    Contrastive Divergence."  Neural Computation 14(8).

Regularisation (optional but recommended)
    Add an L2 penalty on the energy magnitudes to prevent energy from drifting:
    L_reg = lambda * (E(x+)^2 + E(x-)^2).mean()
    Grathwohl et al. (2020) use lambda ~ 1.0.

JEPA prediction loss
---------------------
For I-JEPA and H-JEPA, the predictor is trained to minimise the L2 distance
between its predictions and the (stop-gradient) target encoder representations:
    L = (1/M) * sum_m mean_{n in block_m} || pred_m_n - sg(tgt_m_n) ||_2^2

The target representations are optionally L2-normalised before computing the
loss, which stabilises training (prevents representation collapse to zero).

Reference: Assran et al. (2023). I-JEPA §3, eq. 1.
    https://arxiv.org/abs/2301.08243
"""

from __future__ import annotations
import torch
import torch.nn.functional as F


def contrastive_divergence_loss(
    energy_pos: torch.Tensor,
    energy_neg: torch.Tensor,
    l2_reg_weight: float = 0.0,
) -> torch.Tensor:
    """
    Contrastive Divergence loss for EBM training.

    Parameters
    ----------
    energy_pos    : [B]  energies of real (positive) data samples
    energy_neg    : [B]  energies of SGLD-sampled (negative) samples
    l2_reg_weight : float  coefficient for L2 energy regularisation (default 0)
                    Set to ~1.0 to prevent energy magnitude from exploding.

    Returns
    -------
    loss : scalar torch.Tensor

    Implementation:
    ---------------
    cd_loss  = energy_pos.mean() - energy_neg.mean()
    reg_loss = (energy_pos ** 2 + energy_neg ** 2).mean()
    return cd_loss + l2_reg_weight * reg_loss

    Note: you want energy_pos LOW and energy_neg HIGH, so minimising
    (E_pos - E_neg) achieves that.  The gradient pushes E_pos down and E_neg up.
    """
    raise NotImplementedError


def jepa_prediction_loss(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    normalize_targets: bool = True,
) -> torch.Tensor:
    """
    I-JEPA / H-JEPA prediction loss: mean L2 across all target blocks.

    Parameters
    ----------
    predictions      : list of M tensors, each [B, N_tgt_m, D]
        Predictor outputs for each of the M target blocks.
    targets          : list of M tensors, each [B, N_tgt_m, D]
        Target encoder outputs (should be detached / stop-gradient).
    normalize_targets : bool
        If True, L2-normalise each target token before computing the loss.
        Stabilises training by preventing representations from collapsing to 0.

    Returns
    -------
    loss : scalar torch.Tensor  (mean across blocks and batch)

    Implementation:
    ---------------
    total_loss = 0.0
    for pred, tgt in zip(predictions, targets):
        if normalize_targets:
            tgt = F.normalize(tgt, dim=-1)   # unit-norm along embedding dim
        # MSE over token and embedding dimensions, mean over batch
        block_loss = F.mse_loss(pred, tgt)
        total_loss = total_loss + block_loss
    return total_loss / len(predictions)

    Note: using F.mse_loss (which averages over all elements) is equivalent to
    averaging over patches and embedding dimensions.  Some implementations use
    F.smooth_l1_loss instead for robustness.
    """
    raise NotImplementedError
