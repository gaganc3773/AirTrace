"""Tests for geography, layouts, scenario generation and sensors (no network access)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from airtrace.config import DataConfig, PhysicsConfig, SensorConfig
from airtrace.data.geo import LocalProjection
from airtrace.data.layouts import random_layout
from airtrace.data.raster import polygon_coverage, rasterize_lines
from airtrace.data.sensors import observe, rasterize, sample_sensor_cells
from airtrace.data.synthetic import PROFILES, PRIOR_THRESHOLD, ForcingBuilder, sample_specs, simulate
from airtrace.grid import Grid
from airtrace.solver import ADRSolver, divergence

GRID = Grid(64, 64, 750.0, 750.0)
DATA = DataConfig()
PHYS = PhysicsConfig()


@pytest.fixture(scope="module")
def layouts() -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.stack([random_layout(GRID, rng) for _ in range(3)])


@pytest.fixture(scope="module")
def specs(layouts):
    return sample_specs(np.random.default_rng(1), 16, layouts, GRID, DATA, PHYS)


def test_projection_roundtrip() -> None:
    p = LocalProjection(28.62, 77.15, 48e3, 48e3)
    lon, lat = np.array([77.0, 77.15, 77.3]), np.array([28.5, 28.62, 28.8])
    lo2, la2 = p.to_lonlat(*p.to_xy(lon, lat))
    assert np.allclose(lo2, lon) and np.allclose(la2, lat)
    assert np.allclose(p.to_xy(77.15, 28.62), (24e3, 24e3))


def test_rasterize_lines_conserves_weighted_length() -> None:
    line = np.array([[1000.0, 1000.0], [30000.0, 21000.0], [40000.0, 5000.0]])
    length = np.hypot(29000, 20000) + np.hypot(10000, 16000)
    assert np.isclose(rasterize_lines(GRID, [line], [0.5]).sum(), 0.5 * length, rtol=1e-9)


def test_polygon_coverage_square() -> None:
    sq = np.array([[1500.0, 1500.0], [6000.0, 1500.0], [6000.0, 4500.0], [1500.0, 4500.0]])  # cells 2-7 x 2-5
    cov = polygon_coverage(GRID, [sq])
    assert np.allclose(cov[2:6, 2:8], 1.0) and cov.sum() == pytest.approx(6 * 4)


def test_random_layout_range(layouts) -> None:
    assert layouts.shape == (3, 4, 64, 64)
    assert layouts.min() >= 0 and layouts.max() <= 1
    assert (layouts[:, 0] > 0).mean() > 0.05  # roads exist


def test_profiles_have_unit_daily_mean() -> None:
    assert np.allclose(PROFILES.mean(axis=1), 1.0)
    night = PROFILES[2]
    assert night[8:18].max() == 0.0 and night[22] > 0


def test_specs_deterministic(layouts) -> None:
    a = sample_specs(np.random.default_rng(7), 4, layouts, GRID, DATA, PHYS)
    b = sample_specs(np.random.default_rng(7), 4, layouts, GRID, DATA, PHYS)
    for k, v in a.as_dict().items():
        assert np.array_equal(v, b.as_dict()[k]), k


def test_source_placement_respects_prior(specs, layouts) -> None:
    src = specs.sources
    for b in range(len(specs)):
        for row in src[b][src[b, :, 7] > 0]:
            cat, x, y, off = int(row[0]), row[1], row[2], bool(row[6])
            p = layouts[specs.layout_id[b], cat, int(y // GRID.dy), int(x // GRID.dx)]
            assert (p < PRIOR_THRESHOLD) if off else (p >= PRIOR_THRESHOLD)


def test_daily_emission_matches_inventory(specs, layouts) -> None:
    """Profiles have unit daily mean, so 24 consecutive hours average to the inventory total."""
    fb = ForcingBuilder(specs, layouts, GRID, dtype=torch.float64)
    daily = sum(fb.emission_by_category(h).sum(dim=(1, 2, 3)) for h in range(24)) / 24
    total = specs.traffic_emission + specs.sources[..., 4].sum(axis=1)
    assert np.allclose(daily.numpy(), total, rtol=1e-6)


def test_hidden_night_sources_silent_by_day(layouts) -> None:
    cfg = DATA.model_copy(update={"off_prior_prob": 1.0, "hidden_night_prob": 1.0, "n_construction": (0, 0),
                                  "n_burning": (0, 0)})
    sp = sample_specs(np.random.default_rng(3), 4, layouts, GRID, cfg, PHYS)
    fb = ForcingBuilder(sp, layouts, GRID, dtype=torch.float64)
    lh = sp.local_hours()
    for h in range(sp.wind_u.shape[1]):
        industry = fb.emission_by_category(h)[:, 1].sum(dim=(1, 2)).numpy()
        day = (lh[:, h] >= 8) & (lh[:, h] < 18)
        assert np.all(industry[day] == 0) and np.all(industry[~day & ((lh[:, h] >= 21) | (lh[:, h] < 5))] > 0)


def test_wind_perturbation_divergence_free(specs, layouts) -> None:
    fb = ForcingBuilder(specs, layouts, GRID, dtype=torch.float64)
    f = fb.forcing(5)
    assert divergence(GRID, f.u, f.v).abs().max() < 1e-10


def test_simulate_small_batch(layouts) -> None:
    cfg = DATA.model_copy(update={"sim_hours": 4, "window_hours": 2})
    sp = sample_specs(np.random.default_rng(4), 3, layouts, GRID, cfg, PHYS)
    fb = ForcingBuilder(sp, layouts, GRID, dtype=torch.float64)
    out = simulate(fb, ADRSolver(GRID), window=2, substeps=2)
    assert out["c_hourly"].shape == (3, 2, 64, 64) and out["S_cat"].shape == (3, 4, 64, 64)
    assert torch.isfinite(out["c_hourly"]).all() and out["c_hourly"].min() >= 0 and out["S_cat"].min() >= 0
    # Hourly mean of the final hour lies between its endpoint states' extremes.
    assert out["c_hourly"][:, -1].max() <= torch.maximum(out["c_T"], out["c_Tm1"]).max() + 1e-9


def test_sensor_cells_spacing() -> None:
    cells = sample_sensor_cells(np.random.default_rng(5), 40, GRID, 1500.0)
    assert len({tuple(c) for c in cells}) == 40
    d = np.hypot(*(cells[:, None, :] - cells[None, :, :]).transpose(2, 0, 1)) * 750.0
    assert d[np.triu_indices(40, 1)].min() >= 1500.0


def test_observe_and_rasterize() -> None:
    rng = np.random.default_rng(6)
    c = np.full((6, 64, 64), 100.0)
    cells = sample_sensor_cells(rng, 40, GRID, 1500.0)
    r = observe(rng, c, cells, noise=0.15, dropout=0.25, gap_prob=0.0)
    online = ~np.isnan(r[:, 0])
    assert 0 < online.sum() < 40 and np.all(np.isnan(r[~online]))
    assert abs(np.nanmean(r) / 100.0 - 1) < 0.05  # unbiased multiplicative noise
    values, mask = rasterize(r, cells, GRID)
    assert mask.sum() == np.isfinite(r).sum() and values[mask == 0].max() == 0
    for k in range(6):
        ok = np.isfinite(r[:, k])
        assert np.allclose(values[k, cells[ok, 0], cells[ok, 1]], r[ok, k], rtol=1e-6)


def test_sensor_config_defaults() -> None:
    assert SensorConfig().n_sensors == 40


def test_dataset_roundtrip_resimulates_sources(tmp_path, layouts) -> None:
    """Specs read back from disk rebuild the stored last-hour source field."""
    import json

    from airtrace.data.dataset import SyntheticDataset, write_shard

    cfg = DATA.model_copy(update={"sim_hours": 3, "window_hours": 1})
    sp = sample_specs(np.random.default_rng(8), 4, layouts, GRID, cfg, PHYS)
    out = simulate(ForcingBuilder(sp, layouts, GRID), ADRSolver(GRID), window=1, substeps=2)
    write_shard(tmp_path, 0, {**{k: v.numpy() for k, v in out.items()}, **sp.as_dict()})
    np.save(tmp_path / "layouts.npy", layouts)
    (tmp_path / "meta.json").write_text(json.dumps({"n_scenarios": 4}))

    ds = SyntheticDataset(tmp_path)
    assert len(ds) == 4
    rebuilt = ForcingBuilder(ds.specs([0, 1, 2, 3]), ds.layouts, GRID).source_by_category(2) * 3600.0
    stored = torch.stack([torch.from_numpy(ds[i]["S_cat"].astype(np.float32)) for i in range(4)])
    assert torch.allclose(rebuilt, stored, rtol=2e-3, atol=1e-3)
