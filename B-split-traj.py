"""
Split Japan-wide GPS trajectory logs into per-city subsets.

Input
-----
    /mnt/data/haas/YYYYMMDD.gz                      (one file per day)

Each row is whitespace-separated, no header:
    ID1  id2  lat  lon  unixtime  accuracy  application  date

Example:
    aaaaaa nnnnnn 35.645 138.675 1682477518 36.988 apps.navi 20230426

Output
------
    data/traj_4cities/sim_tokyo_20231001_20231007.csv.gz
    data/traj_4cities/sim_osaka_20231001_20231007.csv.gz
    data/traj_4cities/sim_nagoya_20231001_20231007.csv.gz
    data/traj_4cities/sim_sapporo_20231001_20231007.csv.gz

Each output file is one week (20231001 - 20231007) of trajectories whose
(lat, lon) falls inside that city's bbox. Format matches
sim_nagoya_20230426.csv.gz:
    CSV with header `ID1,id2,lat,lon,unixtime,accuracy,application,date`,
    gzip-compressed.

Usage
-----
    python B-split-traj.py
    python B-split-traj.py --start 20231001 --end 20231007 \\
                           --src-dir /mnt/data/haas \\
                           --out-dir data/traj_4cities
"""
from __future__ import annotations

import argparse
import csv
import gzip
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (north_lat, south_lat, west_lon, east_lon)
CITY_BBOX = {
    "tokyo":   (35.730, 35.640, 139.698, 139.808),
    "osaka":   (34.739, 34.649, 135.448, 135.557),
    "nagoya":  (35.216, 35.126, 136.852, 136.962),
    "sapporo": (43.106, 43.016, 141.295, 141.418),
}

HEADER = ["ID1", "id2", "lat", "lon", "unixtime", "accuracy", "application", "date"]


def city_of(lat: float, lon: float) -> str | None:
    """Return the city key whose bbox contains (lat, lon), or None."""
    for city, (n, s, w, e) in CITY_BBOX.items():
        if s <= lat <= n and w <= lon <= e:
            return city
    return None


def daterange(start: str, end: str):
    """Yield 'YYYYMMDD' strings from start to end (inclusive)."""
    s = date(int(start[:4]), int(start[4:6]), int(start[6:8]))
    e = date(int(end[:4]), int(end[4:6]), int(end[6:8]))
    d = s
    while d <= e:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


def split_traj(src_dir: Path, dates: list[str], out_dir: Path, out_tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    handles: dict[str, "gzip.GzipFile"] = {}
    writers: dict[str, "csv.writer"] = {}
    counts = {city: 0 for city in CITY_BBOX}

    try:
        for city in CITY_BBOX:
            out_path = out_dir / f"sim_{city}_{out_tag}.csv.gz"
            handles[city] = gzip.open(out_path, "wt", encoding="utf-8", newline="")
            writers[city] = csv.writer(handles[city])
            writers[city].writerow(HEADER)

        for yyyymmdd in dates:
            src = src_dir / f"{yyyymmdd}.gz"
            if not src.exists():
                print(f"skip {src} (not found)")
                continue
            n_lines = 0
            n_kept = 0
            with gzip.open(src, "rt", encoding="utf-8", errors="replace") as f:
                for line in f:
                    n_lines += 1
                    parts = line.split()  # any whitespace; ignores blank lines
                    if len(parts) < 8:
                        continue
                    try:
                        lat = float(parts[2])
                        lon = float(parts[3])
                    except ValueError:
                        continue
                    city = city_of(lat, lon)
                    if city is None:
                        continue
                    # Keep original string values to preserve numeric precision.
                    writers[city].writerow(parts[:8])
                    counts[city] += 1
                    n_kept += 1
            print(f"{yyyymmdd}: scanned {n_lines:,}, kept {n_kept:,}")
    finally:
        for h in handles.values():
            h.close()

    print(f"\ntotals across {len(dates)} days:")
    for city, n in counts.items():
        out_name = f"sim_{city}_{out_tag}.csv.gz"
        print(f"  {city:<8s} {n:>12,d}  ->  {out_dir / out_name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=Path("/mnt/data/haas"),
        help="Directory containing YYYYMMDD.gz files.",
    )
    ap.add_argument("--start", default="20231001", help="Start date YYYYMMDD (inclusive).")
    ap.add_argument("--end",   default="20231007", help="End date YYYYMMDD (inclusive).")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "traj_4cities",
        help="Directory to write per-city CSV.gz files into.",
    )
    args = ap.parse_args()

    if not args.src_dir.exists():
        raise SystemExit(f"source dir not found: {args.src_dir}")

    dates = list(daterange(args.start, args.end))
    out_tag = f"{args.start}_{args.end}"
    print(f"days: {dates[0]} .. {dates[-1]}  ({len(dates)} files)")
    split_traj(args.src_dir, dates, args.out_dir, out_tag)


if __name__ == "__main__":
    main()
