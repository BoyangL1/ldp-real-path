#!/usr/bin/env python3
"""
Visualize compact trajectories from `1-split-traj.py` into a single PNG.

Three panels:
  (a) density of every compact point (log)       -- where the users are
  (b) dwell-weighted density (sum of dwell_sec)   -- where they spend time
  (c) every moving trajectory drawn as a polyline -- common corridors
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=Path("data/sim_nagoya_20230426_traj_compact.csv.gz"))
    ap.add_argument("--output", type=Path, default=Path("figs/compact_traj_overview.png"))
    ap.add_argument("--gridsize", type=int, default=150,
                    help="hexbin grid resolution (default: 150)")
    ap.add_argument("--line-alpha", type=float, default=0.03,
                    help="alpha for trajectory polylines (default: 0.03)")
    args = ap.parse_args()

    print(f"[load] {args.input}")
    df = pd.read_csv(args.input)
    print(f"[stats] rows={len(df):,}, users={df['ID1'].nunique():,}, "
          f"trajectories={df.groupby(['ID1','date','traj_id']).ngroups:,}")

    tj_key = ["ID1", "date", "traj_id"]
    df = df.sort_values(tj_key + ["unixtime"])

    segments = [g[["lon", "lat"]].to_numpy()
                for _, g in df.groupby(tj_key, sort=False) if len(g) > 1]
    print(f"[stats] moving trajectories (>=2 cells): {len(segments):,}")

    lon_lo, lon_hi = df["lon"].quantile([0.001, 0.999])
    lat_lo, lat_hi = df["lat"].quantile([0.001, 0.999])
    extent = (lon_lo, lon_hi, lat_lo, lat_hi)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    hb = ax.hexbin(df["lon"], df["lat"], gridsize=args.gridsize, bins="log",
                   cmap="Blues", extent=extent)
    ax.set_title(f"compact points density (log)  n={len(df):,}")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(hb, ax=ax, label="log10(count)")

    ax = axes[1]
    hb = ax.hexbin(df["lon"], df["lat"], C=df["dwell_sec"], reduce_C_function=np.sum,
                   gridsize=args.gridsize, bins="log", cmap="Reds", extent=extent)
    ax.set_title("dwell-weighted density (sum of dwell_sec)")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.colorbar(hb, ax=ax, label="log10(seconds)")

    ax = axes[2]
    lc = LineCollection(segments, colors="steelblue",
                        alpha=args.line_alpha, linewidths=0.3)
    ax.add_collection(lc)
    ax.set_xlim(lon_lo, lon_hi); ax.set_ylim(lat_lo, lat_hi)
    ax.set_title(f"{len(segments):,} moving trajectories")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")

    fig.suptitle(args.input.name, fontsize=11, y=1.02)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[write] {args.output}")


if __name__ == "__main__":
    main()
