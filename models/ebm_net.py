"""
models/ebm_net.py  —  STUB
Energy Network for the EBM.

Maps an image x in R^{C x H x W} to a scalar energy E(x) in R.
Lower energy = higher (unnormalized) probability under the model.

References
----------
Du & Mordatch (2019). "Implicit Generation and Modeling with Energy Based Models."
    NeurIPS 2019.  https://arxiv.org/abs/1903.08689

Grathwohl et al. (2019). "Your Classifier is Secretly an Energy Based Model
    and You Should Treat it Like One."  ICLR 2020.
    https://arxiv.org/abs/1912.03263

Architecture guidance
---------------------
Use strided convolutions to downsample, NOT MaxPool — pooling discards spatial
information that the energy function needs.

Do NOT use BatchNorm anywhere in this network.
BatchNorm recomputes running statistics over the current batch, which corrupts
gradients w.r.t. x during SGLD sampling (the batch statistics change when x
is updated, breaking the Langevin update rule).
Use GroupNorm(num_groups=8, ...) or spectral_norm(nn.Conv2d(...)) instead.

Suggested data flow for 96x96 input with n_channels=[3,64,128,256,512]:
    x  [B, 3, 96, 96]
    -> Conv(3->64,   stride=2)  [B,  64, 48, 48]
    -> Conv(64->128, stride=2)  [B, 128, 24, 24]
    -> Conv(128->256,stride=2)  [B, 256, 12, 12]
    -> Conv(256->512,stride=2)  [B, 512,  6,  6]
    -> GlobalAvgPool            [B, 512]
    -> Linear(512, feature_dim) [B, feature_dim]
    -> Linear(feature_dim, 1)   [B, 1]
    -> squeeze(-1)              [B]
"""

import torch
import torch.nn as nn
from utils.config import EBMConfig


class EnergyNet(nn.Module):
    """
    Convolutional energy function E_theta(x) -> scalar.

    Parameters
    ----------
    cfg : EBMConfig
        cfg.n_channels  : list of ints, e.g. [3, 64, 128, 256, 512]
            n_channels[0] = input channels (3 for RGB)
            n_channels[1:] = output channels of each conv layer
        cfg.feature_dim : int
            Dimension of the linear bottleneck before the final scalar projection.
    """

    def __init__(self, cfg: EBMConfig) -> None:
        super().__init__()

        # TODO: Build self.conv_layers as an nn.Sequential.
        #
        #   For each consecutive pair in cfg.n_channels, create:
        #     nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)
        #     nn.GroupNorm(num_groups=min(8, out_c), num_channels=out_c)
        #     nn.LeakyReLU(0.2, inplace=True)
        #
        #   Hint: use zip(cfg.n_channels[:-1], cfg.n_channels[1:])

        # TODO: Build self.head as an nn.Sequential:
        #   nn.AdaptiveAvgPool2d(1)  — collapses spatial dims to [B, C, 1, 1]
        #   nn.Flatten()             — -> [B, C]
        #   nn.Linear(cfg.n_channels[-1], cfg.feature_dim)
        #   nn.LeakyReLU(0.2)
        #   nn.Linear(cfg.feature_dim, 1)

        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute energy scores for a batch of images.

        Parameters
        ----------
        x : torch.Tensor  shape [B, C, H, W]
            Normalized images OR in-progress SGLD samples.
            Values are NOT clipped — the network must handle any float range.

        Returns
        -------
        energy : torch.Tensor  shape [B]
            Scalar energy per image.  Lower energy = more plausible under model.

        Note: the output is an UNnormalized log-density (free energy).
        The partition function Z is never computed explicitly.
        """
        # TODO:
        #   out = self.conv_layers(x)
        #   out = self.head(out)          # [B, 1]
        #   return out.squeeze(-1)        # [B]
        raise NotImplementedError
