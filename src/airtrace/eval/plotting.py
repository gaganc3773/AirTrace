"""Shared matplotlib style so every report figure looks the same."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f", "#9c6644"]
FIELD_CMAP = "magma_r"


def set_style() -> None:
    """Apply the AirTrace figure style globally."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "legend.frameon": False,
            "image.cmap": FIELD_CMAP,
        }
    )


def savefig(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Save ``fig`` as PNG under ``out_dir`` and close it."""
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
