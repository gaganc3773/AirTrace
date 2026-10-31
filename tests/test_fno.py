"""Tests for the FNO architecture and its input pipeline."""

from __future__ import annotations

import numpy as np
import torch

from airtrace.config import FNOConfig
from airtrace.grid import Grid
from airtrace.models.fno import FNO2d, SpectralConv2d
from airtrace.models.fno_data import N_IN, Met, build_inputs, rasterize_network, sample_networks
from airtrace.models.fno_train import activate, make_model

GRID = Grid(64, 64, 750.0, 750.0)


def _met(B: int) -> Met:
    z = torch.zeros(B, 6)
    return Met(z + 2.0, z - 1.0, z + 0.3, torch.randn(B, 2, 64, 64), z + 500.0, torch.arange(B) % 24,
               torch.rand(B, 4, 64, 64))


def test_spectral_conv_is_translation_equivariant() -> None:
    """A periodic shift of the input shifts the output (global convolution)."""
    torch.manual_seed(0)
    sc = SpectralConv2d(3, 4, 6)
    x = torch.randn(1, 3, 32, 32)
    a = torch.roll(sc(x), shifts=(5, 7), dims=(-2, -1))
    b = sc(torch.roll(x, shifts=(5, 7), dims=(-2, -1)))
    assert torch.allclose(a, b, atol=1e-5)


def test_fno_shapes_and_activation() -> None:
    m = make_model(FNOConfig(width=16, modes=8, layers=2))
    out = activate(m(torch.randn(2, N_IN, 64, 64)))
    assert out.shape == (2, 11, 64, 64) and (out >= 0).all()
    assert torch.allclose(out[:, 8], out[:, 8, :1, :1].expand(-1, 64, 64))  # background is uniform


def test_fno_padding_stops_wraparound() -> None:
    """A localised input perturbation barely affects the opposite edge (zero padding)."""
    torch.manual_seed(1)
    m = FNO2d(1, 1, width=8, modes=8, layers=2, pad=16)
    x = torch.zeros(1, 1, 64, 64)
    x2 = x.clone()
    x2[..., 30:34, 0:2] = 10.0
    d = (m(x2) - m(x)).abs()[0, 0]
    assert d[:, -3:].max() < 0.5 * d[:, :3].max()


def test_sample_networks_counts_and_noise() -> None:
    gen = torch.Generator().manual_seed(2)
    c = torch.full((8, 6, 64, 64), 100.0)
    vals, mask = sample_networks(c, (5, 40), (0.1, 0.1), (0.0, 0.0), 0.0, gen)
    per = mask[:, 0].sum(dim=(1, 2))
    assert per.min() >= 5 and per.max() <= 40
    assert torch.equal(mask[:, 0], mask[:, 5])  # same network every hour when there are no gaps
    obs = vals[mask > 0]
    assert abs(float(obs.mean()) / 100 - 1) < 0.05 and float(obs.std()) / 100 > 0.05


def test_build_inputs_scale_is_median_reading() -> None:
    vals = torch.zeros(2, 6, 64, 64)
    mask = torch.zeros_like(vals)
    vals[:, :, 10, 10], vals[:, :, 20, 20], vals[:, :, 30, 30] = 40.0, 80.0, 400.0
    mask[:, :, 10, 10] = mask[:, :, 20, 20] = mask[:, :, 30, 30] = 1.0
    x, s = build_inputs(vals, mask, _met(2))
    assert x.shape == (2, N_IN, 64, 64) and torch.allclose(s, torch.tensor([80.0, 80.0]))
    assert torch.isclose(x[0, 0, 20, 20], torch.tensor(1.0)) and x[0, 0, 0, 0] == 0


def test_rasterize_network_skips_nan() -> None:
    cells = np.array([[1, 2], [3, 4]], dtype=np.int16)
    r = np.array([[10.0, np.nan], [5.0, 6.0]])
    v, m = rasterize_network(cells, r, GRID)
    assert v[0, 1, 2] == 10 and m[1, 1, 2] == 0 and v[1, 3, 4] == 6 and m.sum() == 3
