"""Fourier Neural Operator (Li et al., 2021) for 2D fields, implemented directly in PyTorch."""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv2d(nn.Module):
    """Global convolution: multiply the lowest ``modes x modes`` Fourier coefficients by learned weights."""

    def __init__(self, c_in: int, c_out: int, modes: int):
        super().__init__()
        self.c_out, self.modes = c_out, modes
        scale = 1.0 / (c_in * c_out)
        self.w_pos = nn.Parameter(scale * torch.randn(c_in, c_out, modes, modes, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.randn(c_in, c_out, modes, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        m = self.modes
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.c_out, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w_pos)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w_neg)
        return torch.fft.irfft2(out, s=(H, W))


class FNO2d(nn.Module):
    """Lift → ``layers`` x (spectral conv + pointwise conv, GELU) → project.

    The input is zero-padded by ``pad`` cells so the FFT's implied periodicity
    does not wrap plumes across the (non-periodic) domain edges.
    """

    def __init__(self, c_in: int, c_out: int, width: int = 48, modes: int = 12, layers: int = 4, pad: int = 8):
        super().__init__()
        self.pad = pad
        self.lift = nn.Conv2d(c_in, width, 1)
        self.spectral = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, c_out, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        x = nn.functional.pad(x, (0, self.pad, 0, self.pad))
        for k, (sp, pw) in enumerate(zip(self.spectral, self.pointwise)):
            x = sp(x) + pw(x)
            if k < len(self.spectral) - 1:
                x = nn.functional.gelu(x)
        x = x[..., : x.shape[-2] - self.pad, : x.shape[-1] - self.pad]
        return self.proj(x)
