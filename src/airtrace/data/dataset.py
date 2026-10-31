"""On-disk synthetic dataset: shards of ``.npy`` files, memory-mapped on read.

Layout::

    <root>/meta.json
    <root>/layouts.npy                 (L, 4, ny, nx) prior maps, index 0 = Delhi
    <root>/shard_000/<key>.npy         one file per field, leading dim = scenarios

Fields (float16 unless noted): ``c_hourly`` (W, ny, nx), ``c_T``, ``c_Tm1``
(ny, nx), ``S_cat`` (4, ny, nx) in µg m^-3 h^-1, ``sensor_cells`` (n, 2) int16,
``sensor_readings`` (n, W) float32 with NaN gaps, ``sensor_noise`` float32,
plus every :class:`ScenarioSpecs` field (float64/int64) for exact re-simulation.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np

from airtrace.data.synthetic import ScenarioSpecs

FIELD_DTYPES = {
    "c_hourly": np.float16,
    "c_T": np.float16,
    "c_Tm1": np.float16,
    "S_cat": np.float16,
    "sensor_cells": np.int16,
    "sensor_readings": np.float32,
    "sensor_noise": np.float32,
}
SPEC_KEYS = tuple(f.name for f in fields(ScenarioSpecs))


def write_shard(root: Path, index: int, arrays: dict[str, np.ndarray]) -> Path:
    """Write one shard; field arrays are cast to their storage dtype."""
    d = root / f"shard_{index:03d}"
    d.mkdir(parents=True, exist_ok=True)
    for key, arr in arrays.items():
        np.save(d / f"{key}.npy", np.asarray(arr).astype(FIELD_DTYPES.get(key, np.asarray(arr).dtype), copy=False))
    return d


class SyntheticDataset:
    """Read-only, lazily memory-mapped view of a generated dataset."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.layouts = np.load(self.root / "layouts.npy")
        self.shards = sorted(p for p in self.root.glob("shard_*") if p.is_dir())
        sizes = [np.load(p / "layout_id.npy", mmap_mode="r").shape[0] for p in self.shards]
        self._offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.keys = tuple(sorted(p.stem for p in self.shards[0].glob("*.npy"))) if self.shards else ()
        self._cache: dict[tuple[int, str], np.ndarray] = {}

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _array(self, shard: int, key: str) -> np.ndarray:
        if (shard, key) not in self._cache:
            self._cache[(shard, key)] = np.load(self.shards[shard] / f"{key}.npy", mmap_mode="r")
        return self._cache[(shard, key)]

    def _locate(self, i: int) -> tuple[int, int]:
        if not 0 <= i < len(self):
            raise IndexError(i)
        s = int(np.searchsorted(self._offsets, i, side="right") - 1)
        return s, i - int(self._offsets[s])

    def get(self, i: int, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        """All (or the selected) arrays of scenario ``i``."""
        s, j = self._locate(i)
        keys = keys or self.keys
        return {k: np.asarray(self._array(s, k)[j]) for k in keys}

    __getitem__ = get

    def specs(self, indices: list[int] | np.ndarray) -> ScenarioSpecs:
        """Scenario specs for re-simulation."""
        rows = [self.get(int(i), SPEC_KEYS) for i in indices]
        return ScenarioSpecs(**{k: np.stack([r[k] for r in rows]) for k in SPEC_KEYS})
