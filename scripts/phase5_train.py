"""Phase 5: train the FNO deep ensemble.

Usage:
    python scripts/phase5_train.py                       # 5 members, full data
    python scripts/phase5_train.py --trial 3 --epochs 3  # quick check on the first 3 shards
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from airtrace.config import load_config
from airtrace.data.dataset import SyntheticDataset
from airtrace.grid import Grid
from airtrace.models.fno_data import S_SLICE, GPUData, build_inputs
from airtrace.models.fno_train import physics_residual, save_member, stored_networks, train_member


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--members", type=int, default=None)
    parser.add_argument("--start", type=int, default=0, help="first member index (resume an ensemble)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--trial", type=int, default=0, help="use only the first N shards (smoke test)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override FNO config fields")
    args = parser.parse_args()
    cfg = load_config(args.config)
    over = {k: float(v) for k, v in (kv.split("=") for kv in args.set)}
    if args.epochs:
        over["epochs"] = args.epochs
    if args.trial:
        n = args.trial * cfg.data.shard_size
        over.update(train_range=(0, int(n * 0.85)), val_range=(int(n * 0.85), n))
    fc = cfg.fno.model_validate({**cfg.fno.model_dump(), **over})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    grid = Grid.from_config(cfg.domain)
    ds = SyntheticDataset(cfg.path("artifacts") / "datasets" / cfg.data.name)
    data = GPUData(ds, cfg.data.window_hours, dev, n_shards=args.trial or None)
    print(f"loaded {data.n} scenarios on {dev}; train {fc.train_range}, val {fc.val_range}")

    val_idx = np.arange(*fc.val_range)
    vals, masks = stored_networks(ds, val_idx, grid, dev)
    val = (torch.as_tensor(val_idx, device=dev), vals, masks)

    # How consistent is the hour-integrated residual on the *true* fields? (sets the scale of w_phys)
    i = val[0][:64]
    x, s = build_inputs(vals[:64], masks[:64], data.met(i))
    y = data.targets(i, s)
    r = physics_residual(y, data, i, s, grid)
    print(f"truth physics residual RMS {float(r.pow(2).mean().sqrt()):.3f} vs source RMS "
          f"{float(y[:, S_SLICE].sum(1).pow(2).mean().sqrt()):.3f} (scaled units)")

    out_dir = cfg.path("artifacts") / ("fno_trial" if args.trial else "fno")
    out_dir.mkdir(exist_ok=True)
    hist_path = out_dir / "histories.json"
    histories = json.loads(hist_path.read_text()) if args.start and hist_path.exists() else {}
    for k in range(args.start, args.members or fc.members):
        print(f"member {k}")
        model, hist = train_member(data, fc, grid, seed=cfg.seed + 1000 * k, val=val)
        save_member(model, fc, out_dir / f"member_{k}.pt", hist)
        histories[str(k)] = hist
        hist_path.write_text(json.dumps(histories, indent=2))


if __name__ == "__main__":
    main()
