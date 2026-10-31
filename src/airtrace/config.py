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
    wind_speed_range: tuple[float, float] = (0.3, 6.0)
    mixing_height_night: float = 250.0
    mixing_height_day: float = 1200.0
    background_range: tuple[float, float] = (20.0, 250.0)


class SolverConfig(BaseModel):
    """Numerical settings of the finite-volume solver."""

    limiter: Literal["koren", "mc", "vanleer", "linear3", "upwind1"] = "koren"
    dt_max: float = 60.0
    cfl_max: float = 0.5
    forcing_interval: float = 600.0


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
