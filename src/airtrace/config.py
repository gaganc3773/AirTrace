"""Typed, YAML-backed configuration and project path resolution."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class DomainConfig(BaseModel):
    """Horizontal model domain (a local metric box around a city centre)."""

    name: str = "delhi"
    center_lat: float = 28.62
    center_lon: float = 77.15
    width_m: float = 48000.0
    height_m: float = 48000.0
    nx: int = 64
    ny: int = 64


class PhysicsConfig(BaseModel):
    """Physical constants and the ranges sampled for synthetic scenarios."""

    K: float = 75.0
    lam: float = 2.0e-5
    K_range: tuple[float, float] = (30.0, 150.0)
    lam_range: tuple[float, float] = (1.0e-5, 5.0e-5)
    wind_speed_range: tuple[float, float] = (0.5, 6.0)
    mixing_height_night: float = 250.0
    mixing_height_day: float = 1200.0
    background_range: tuple[float, float] = (15.0, 150.0)


class SolverConfig(BaseModel):
    """Numerical settings of the finite-volume solver."""

    limiter: Literal["koren", "mc", "vanleer", "linear3", "upwind1"] = "koren"
    dt_max: float = 60.0
    cfl_max: float = 0.5
    forcing_interval: float = 600.0


class SensorConfig(BaseModel):
    """Synthetic sensor network: placement and measurement error."""

    n_sensors: int = 40
    min_dist_m: float = 1500.0
    noise_range: tuple[float, float] = (0.10, 0.20)  # multiplicative lognormal sigma
    dropout_range: tuple[float, float] = (0.0, 0.3)  # fraction of sensors offline for the window
    hourly_gap_prob: float = 0.05


class DataConfig(BaseModel):
    """Synthetic scenario generation."""

    name: str = "synth_v1"
    n_scenarios: int = 20000
    shard_size: int = 1000
    batch_size: int = 500
    sim_hours: int = 30
    window_hours: int = 6
    delhi_fraction: float = 0.7
    n_random_layouts: int = 256
    total_emission_kg_s: tuple[float, float] = (1.0, 6.0)
    max_source_g_s: tuple[float, float, float] = (150.0, 60.0, 200.0)  # industry, construction, burning
    category_alpha: tuple[float, float, float, float] = (2.0, 2.0, 1.5, 1.2)
    n_industry: tuple[int, int] = (3, 12)
    n_construction: tuple[int, int] = (0, 6)
    n_burning: tuple[int, int] = (0, 5)
    off_prior_prob: float = 0.15
    hidden_night_prob: float = 0.5
    nw_wind_prob: float = 0.5
    wind_pert_frac: tuple[float, float] = (0.0, 0.3)
    sensors: SensorConfig = SensorConfig()


class PathsConfig(BaseModel):
    """Output locations, relative to the project root."""

    artifacts: str = "artifacts"
    figures: str = "reports/figures"


class Config(BaseModel):
    """Top-level AirTrace configuration."""

    seed: int = 42
    domain: DomainConfig = DomainConfig()
    physics: PhysicsConfig = PhysicsConfig()
    solver: SolverConfig = SolverConfig()
    data: DataConfig = DataConfig()
    paths: PathsConfig = PathsConfig()

    def path(self, key: Literal["artifacts", "figures"]) -> Path:
        """Absolute path for a configured output directory, created if missing."""
        p = project_root() / getattr(self.paths, key)
        p.mkdir(parents=True, exist_ok=True)
        return p


def project_root() -> Path:
    """Project root: ``$AIRTRACE_ROOT`` if set, else the nearest parent holding pyproject.toml."""
    env = os.environ.get("AIRTRACE_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load a YAML config; defaults to ``configs/default.yaml`` under the project root."""
    path = Path(path) if path is not None else project_root() / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
