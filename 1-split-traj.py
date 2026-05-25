#!/usr/bin/env python3
"""
Split per-user GPS traces in `sim_nagoya_YYYYMMDD.csv.gz` into trajectories
by long-stay detection. Writes two files:

  <input>_traj.csv.gz
      every original row plus a `traj_id` column (k-th trajectory of each
      (ID1, date), 1-indexed).

  <input>_traj_compact.csv.gz
      consecutive rows at the same (lat, lon) *within one trajectory* merged
      into a single row, then filtered to keep only MOVING trajectories
      (>= 2 compact cells). Schema:
          ID1, id2, lat, lon, unixtime, accuracy, application, date, traj_id,
          end_unixtime, dwell_sec, n_pings
      where `unixtime` is the segment start, `end_unixtime` the segment end,
      `dwell_sec = end_unixtime - unixtime`, `n_pings` the number of merged
      source rows, `accuracy` the segment mean, `application` the first row's.

Why this design?  sim_nagoya_20230426.csv.gz has:
  * lat/lon quantized to 0.001 deg (~100 m); 80% of consecutive in-user steps
    have zero displacement -> no GPS jitter to smooth.
  * unixtime regularly sampled at ~100 s (99% of gaps in [60, 180] s).
  * ~8% of location-runs dwell >= 10 min but cover ~70% of all rows --
    a few real stay points dominate time, with many brief grid crossings
    in between.
So a grid/anchor port of 0-split-traj.ipynb is the wrong shape for this data.
The only knob that matters is how long a stay must be to split trajectories:
that's `--min-stay-sec` (default 600 = 10 min).

Algorithm (one pass):
  1. sort by (ID1, date, unixtime)
  2. consecutive identical (lat, lon) rows within same (ID1, date) -> one
     location-run
  3. each run's dwell = last_unixtime - first_unixtime
  4. runs with dwell >= min_stay_sec are "long stays"
  5. a new trajectory starts at a row that is (a) the first row of its
     (ID1, date), OR (b) the *last* ping of a long-stay run that is
     followed by more data in the same (ID1, date). In other words, the
     last ping of a long stay is the "departure moment" and becomes the
     starting anchor of the next trajectory. traj_id = cumulative count
     of such starts, 1-indexed.
  6. every row gets a traj_id. All but the last ping of a long stay belong
     to the trajectory whose final rest the stay is; the last ping of that
     stay belongs to the next trajectory (its starting point). In the
     compact form this splits a long-stay run into two rows at the same
     (lat, lon): the rest portion (n_pings=N-1, dwell>=min_stay_sec,
     traj_id=k) and the single departure ping (n_pings=1, dwell=0,
     traj_id=k+1). Movement between stays stays entirely in one traj.

On sim_nagoya_20230426 with min_stay_sec=600 this yields ~14 trajectories
per (user, day) (median 12), and the compact form is ~5x shorter than the
per-ping form because ~80% of in-user steps are zero-displacement.

Usage:
    python 1-split-traj.py
    python 1-split-traj.py --min-stay-sec 900
    python 1-split-traj.py --input data/sim_nagoya_20230426.csv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["ID1", "lat", "lon", "unixtime"]


def prepare(df: pd.DataFrame, min_stay_sec: int) -> pd.DataFrame:
    """Sort df, attach `loc_run_id` (consecutive same-coord group, global id)
    and `traj_id` (k-th trajectory of (ID1, date)). The last ping of a long
    stay becomes the starting point of the next trajectory."""
    df = df.sort_values(["ID1", "date", "unixtime"], kind="mergesort").reset_index(drop=True)

    first_in_group = (df["ID1"] != df["ID1"].shift()) | (df["date"] != df["date"].shift())
    loc_change = (
        first_in_group
        | (df["lat"] != df["lat"].shift())
        | (df["lon"] != df["lon"].shift())
    )
    df["loc_run_id"] = loc_change.cumsum().astype(np.int64)

    run_stats = (
        df.groupby("loc_run_id", sort=True)
          .agg(start_ut=("unixtime", "min"), end_ut=("unixtime", "max"))
    )
    run_stats["is_long_stay"] = (run_stats["end_ut"] - run_stats["start_ut"]) >= min_stay_sec

    is_long_row = df["loc_run_id"].map(run_stats["is_long_stay"]).fillna(False).to_numpy()
    next_run    = df["loc_run_id"].shift(-1)
    is_last_of_run = (df["loc_run_id"].to_numpy() != next_run.to_numpy())
    has_next_in_group = (
        (df["ID1"] == df["ID1"].shift(-1))
        & (df["date"] == df["date"].shift(-1))
    ).fillna(False).to_numpy()
    is_departure_ping = is_long_row & is_last_of_run & has_next_in_group

    is_traj_start = first_in_group.to_numpy() | is_departure_ping

    df["traj_id"] = (
        pd.Series(is_traj_start.astype(np.int32))
          .groupby([df["ID1"], df["date"]], sort=False)
          .cumsum()
          .astype(np.int32)
          .to_numpy()
    )
    return df


def compact(df_with_runs: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive same-(lat, lon) rows within one trajectory into one row.

    A long-stay run whose last ping was reassigned to the next trajectory is
    split into two compact rows at the same (lat, lon): the rest portion
    (traj_id=k, n_pings>=1) and the departure ping (traj_id=k+1, n_pings=1).
    """
    grp = df_with_runs.groupby(["loc_run_id", "traj_id"], sort=True)
    out = grp.agg(
        ID1=("ID1", "first"),
        id2=("id2", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
        unixtime=("unixtime", "min"),
        end_unixtime=("unixtime", "max"),
        accuracy=("accuracy", "mean"),
        application=("application", "first"),
        date=("date", "first"),
        n_pings=("unixtime", "size"),
    ).reset_index()
    out["dwell_sec"] = (out["end_unixtime"] - out["unixtime"]).astype(np.int64)
    out["n_pings"] = out["n_pings"].astype(np.int32)
    return out[[
        "ID1", "id2", "lat", "lon", "unixtime", "accuracy", "application",
        "date", "traj_id", "end_unixtime", "dwell_sec", "n_pings",
    ]]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", type=Path, default=Path("data/sim_nagoya_20230426.csv.gz"))
    ap.add_argument("--output", type=Path, default=None,
                    help="per-ping output path. default: <input>_traj.csv.gz")
    ap.add_argument("--output-compact", type=Path, default=None,
                    help="compact output path. default: <input>_traj_compact.csv.gz")
    ap.add_argument("--min-stay-sec", type=int, default=600,
                    help="a location-run this long (seconds) counts as a stay that ends "
                         "the current trajectory (default: 600)")
    args = ap.parse_args()

    if args.output is None:
        args.output = args.input.with_name(args.input.name.replace(".csv.gz", "_traj.csv.gz"))
    if args.output_compact is None:
        args.output_compact = args.input.with_name(args.input.name.replace(".csv.gz", "_traj_compact.csv.gz"))

    print(f"[load] {args.input}")
    df = pd.read_csv(args.input, compression="gzip")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["unixtime"], unit="s").dt.strftime("%Y%m%d").astype(np.int64)

    print(f"[stats] rows={len(df):,}, users={df['ID1'].nunique():,}, days={df['date'].nunique()}")

    with_runs = prepare(df, min_stay_sec=args.min_stay_sec)
    per_ping = with_runs.drop(columns=["loc_run_id"])
    compact_df = compact(with_runs)

    per_ud = per_ping.groupby(["ID1", "date"])["traj_id"].nunique()
    sizes = per_ping.groupby(["ID1", "date", "traj_id"]).size()
    print(
        f"[traj] per (user,day): mean={per_ud.mean():.2f}, median={per_ud.median():.0f}, "
        f"min={per_ud.min()}, max={per_ud.max()}  (threshold={args.min_stay_sec}s)"
    )
    print(
        f"[traj] pings per trajectory: mean={sizes.mean():.1f}, median={sizes.median():.0f}, "
        f"p95={sizes.quantile(0.95):.0f}, max={sizes.max()}"
    )
    print(
        f"[compact] rows before filter: {len(compact_df):,}  "
        f"({len(compact_df)/len(per_ping)*100:.1f}% of per-ping)"
    )

    # Keep only MOVING trajectories (>= 2 compact cells).
    tj_key = ["ID1", "date", "traj_id"]
    n_cells = compact_df.groupby(tj_key)["lat"].transform("size")
    n_traj_before = compact_df[tj_key].drop_duplicates().shape[0]
    compact_df = compact_df[n_cells >= 2].reset_index(drop=True)
    n_traj_after = compact_df[tj_key].drop_duplicates().shape[0]
    print(
        f"[compact] moving-only filter (n_cells>=2): "
        f"trajectories {n_traj_before:,} -> {n_traj_after:,}; "
        f"rows {len(compact_df):,}"
    )
    print(
        f"[compact] dwell_sec: mean={compact_df['dwell_sec'].mean():.0f}, "
        f"median={compact_df['dwell_sec'].median():.0f}, max={compact_df['dwell_sec'].max()}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {args.output}")
    per_ping.to_csv(args.output, index=False, compression="gzip")
    print(f"[write] {args.output_compact}")
    compact_df.to_csv(args.output_compact, index=False, compression="gzip")
    print("[done]")


if __name__ == "__main__":
    main()
