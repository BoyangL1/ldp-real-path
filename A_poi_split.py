"""
Split a country-wide POI TSV (loco format) into per-city TSVs by bounding box.

Input
-----
    /mnt/data/poi/shop_info_genre.tsv  (on the GPU server)

Each row is tab-separated, in the same shape as data/poi_nagoya_loco.tsv:
    <row_id>\t<poi_id>\t<name>\t<genre_json>\t<coord_json>
where <coord_json> is "[lon, lat]" (lon first, lat second).

Output
------
    data/poi_4cities/poi_tokyo_loco.tsv
    data/poi_4cities/poi_osaka_loco.tsv
    data/poi_4cities/poi_nagoya_loco.tsv
    data/poi_4cities/poi_sapporo_loco.tsv

Each output row is the verbatim input row (same columns, same order) so the
existing readers (0-grid_privacy.py et al.) can consume them unchanged.

Usage
-----
    python A_poi_split.py
    python A_poi_split.py --src /mnt/data/poi/shop_info_genre.tsv \\
                          --out-dir data/poi_4cities
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (north_lat, south_lat, west_lon, east_lon)
CITY_BBOX = {
    "tokyo":   (35.730, 35.640, 139.698, 139.808),
    "osaka":   (34.739, 34.649, 135.448, 135.557),
    "nagoya":  (35.216, 35.126, 136.852, 136.962),
    "sapporo": (43.106, 43.016, 141.295, 141.418),
}


def parse_coord(coord_field: str) -> tuple[float, float] | None:
    """Return (lon, lat) parsed from a "[lon,lat]" JSON-ish string, or None."""
    s = coord_field.strip()
    if not s or s[0] != "[":
        return None
    try:
        lon, lat = json.loads(s)
        return float(lon), float(lat)
    except (ValueError, TypeError):
        return None


def city_of(lon: float, lat: float) -> str | None:
    """Return the city key whose bbox contains (lon, lat), or None."""
    for city, (n, s, w, e) in CITY_BBOX.items():
        if s <= lat <= n and w <= lon <= e:
            return city
    return None


def split_poi(src: Path, out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        city: (out_dir / f"poi_{city}_loco.tsv").open("w", encoding="utf-8")
        for city in CITY_BBOX
    }
    counts = {city: 0 for city in CITY_BBOX}
    total = 0
    bad = 0
    try:
        with src.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total += 1
                # Last tab-separated field is the coord JSON; rsplit avoids
                # splitting the (possibly tab-free) genre JSON unnecessarily.
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    bad += 1
                    continue
                coord = parse_coord(parts[-1])
                if coord is None:
                    bad += 1
                    continue
                lon, lat = coord
                city = city_of(lon, lat)
                if city is None:
                    continue
                handles[city].write(line if line.endswith("\n") else line + "\n")
                counts[city] += 1
    finally:
        for h in handles.values():
            h.close()

    print(f"scanned {total:,} rows, {bad:,} malformed")
    for city, n in counts.items():
        print(f"  {city:<8s} {n:>10,d}  ->  {out_dir / f'poi_{city}_loco.tsv'}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        type=Path,
        default=Path("/mnt/data/poi/shop_info_genre.tsv"),
        help="Path to the country-wide POI TSV.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "poi_4cities",
        help="Directory to write per-city TSVs into.",
    )
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"source POI file not found: {args.src}")
    split_poi(args.src, args.out_dir)


if __name__ == "__main__":
    main()
