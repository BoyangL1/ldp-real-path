"""
Grid-level privacy scoring for a POI dataset (Nagoya by default).

Pipeline (same shape as 1-grid-privacy.ipynb, but for a single city):
    1. Read POI TSV (loco format), extract (lon, lat) and level-2 genre codes.
    2. Tile the bounding box into CELL_M x CELL_M cells. x=col (lon), y=row (lat).
    3. Aggregate POI counts per (cell, level-2 category).
    4. Privacy score per cell = weighted mean of per-category privacy scores
       from a JSON table, optionally weighted by an HHI concentration term.
    5. Save outputs (CSV + .npy matrices) and render 2 heatmaps:
       POI density and privacy score.

All parameters are configurable via command-line flags; defaults are the
values we used for the first Nagoya pass.

Examples
--------
    # default Nagoya run
    python 0-grid_privacy.py

    # override bbox + cell size + output dir
    python 0-grid_privacy.py \\
        --min-lon 136.85 --max-lon 136.98 \\
        --min-lat 35.10  --max-lat 35.25 \\
        --cell-m 200 \\
        --out-dir data/privacy_outputs/nagoya_200m
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parent


# Default values (used when the CLI flag is omitted). Changing them here
# also changes the CLI default.
DEFAULTS = dict(
    tsv=ROOT / "data/poi_nagoya_loco.tsv",
    score_json=ROOT / "data/genre_privacy_score.json",
    out_dir=ROOT / "data/privacy_outputs/nagoya",
    fig_dir=ROOT / "figs",
    fig_prefix="grid",
    min_lon=136.8516, max_lon=136.9616,
    min_lat=35.1365,  max_lat=35.2265,
    cell_m=100.0,
    alpha=0.5,
    use_conc=True,
)


def _set_jp_font():
    for f in ("Noto Sans CJK JP", "IPAexGothic", "IPAGothic",
              "TakaoPGothic", "Hiragino Sans", "Yu Gothic", "Meiryo"):
        plt.rcParams["font.family"] = f
        return


# --- 1. Load POIs -----------------------------------------------------------

def load_pois(tsv: Path) -> pd.DataFrame:
    rows = []
    for line in Path(tsv).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        poi_id, name, codes_json, coord_json = line.split("\t")
        codes = json.loads(codes_json)
        lon, lat = json.loads(coord_json)
        # A POI may carry >1 level-3 code. Collapse to level-2 (first 4 chars)
        # and de-duplicate so a single POI cannot double-count a category.
        cat2 = {c["Code"][:4] for c in codes}
        rows.append({"id": poi_id, "name": name, "lon": lon, "lat": lat,
                     "cats2": sorted(cat2)})
    return pd.DataFrame(rows)


def load_scores(path: Path) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {c["code2"]: c for c in data["categories"]}


# --- 2. Grid construction ---------------------------------------------------

def grid_spec(min_lon, max_lon, min_lat, max_lat, cell_m):
    center_lat = 0.5 * (min_lat + max_lat)
    d_lat = cell_m / 111_000.0
    d_lon = cell_m / (111_000.0 * math.cos(math.radians(center_lat)))
    nx = int(math.ceil((max_lon - min_lon) / d_lon))
    ny = int(math.ceil((max_lat - min_lat) / d_lat))
    return dict(min_lon=min_lon, max_lon=max_lon,
                min_lat=min_lat, max_lat=max_lat,
                d_lon=d_lon, d_lat=d_lat, nx=nx, ny=ny, cell_m=cell_m)


def to_xy(lon, lat, gs):
    x = np.floor((lon - gs["min_lon"]) / gs["d_lon"]).astype(int)
    y = np.floor((lat - gs["min_lat"]) / gs["d_lat"]).astype(int)
    x = np.clip(x, 0, gs["nx"] - 1)
    y = np.clip(y, 0, gs["ny"] - 1)
    return x, y


# --- 3. Aggregate: (grid, category) -> POI count ---------------------------

def build_count_matrix(poi_df: pd.DataFrame, gs: dict, cat_ids: list[str]):
    cat_idx = {c: i for i, c in enumerate(cat_ids)}
    exp = poi_df.explode("cats2").rename(columns={"cats2": "code2"})
    exp = exp[exp["code2"].isin(cat_idx)].copy()

    # Drop POIs that fall outside the bounding box entirely.
    in_box = ((exp["lon"] >= gs["min_lon"]) & (exp["lon"] < gs["max_lon"]) &
              (exp["lat"] >= gs["min_lat"]) & (exp["lat"] < gs["max_lat"]))
    exp = exp[in_box].copy()

    xs, ys = to_xy(exp["lon"].to_numpy(), exp["lat"].to_numpy(), gs)
    exp["grid_flat"] = ys * gs["nx"] + xs
    exp["cat_idx"] = exp["code2"].map(cat_idx)

    agg = (exp.groupby(["grid_flat", "cat_idx"])
              .size().rename("n").reset_index())
    n_grid = gs["nx"] * gs["ny"]
    M = sparse.csr_matrix(
        (agg["n"].to_numpy(np.float32),
         (agg["grid_flat"].to_numpy(), agg["cat_idx"].to_numpy())),
        shape=(n_grid, len(cat_ids)),
    )
    return M


# --- 4. Privacy score per cell ---------------------------------------------

def compute_privacy(M: sparse.csr_matrix, cat_scores: np.ndarray,
                    alpha: float, use_conc: bool):
    """
    p(cat|g)        = count[g,cat] / total[g]
    privacy_mean[g] = Σ p(cat|g) * score[cat]
    hhi_norm[g]     = normalized Herfindahl index of p(.|g)
    privacy[g]      = privacy_mean[g] * (alpha + (1-alpha)*hhi_norm[g])
                      if use_conc else privacy_mean[g]
    """
    row_sum = np.asarray(M.sum(axis=1)).ravel()
    safe = np.where(row_sum > 0, row_sum, 1.0)
    P = sparse.diags(1.0 / safe) @ M

    privacy_mean = P.dot(cat_scores)

    sq = P.multiply(P)
    hhi = np.asarray(sq.sum(axis=1)).ravel()
    k = M.shape[1]
    hhi_norm = np.clip((hhi - 1.0 / k) / (1.0 - 1.0 / k), 0.0, 1.0) if k > 1 else np.ones_like(hhi)

    if use_conc:
        privacy = privacy_mean * (alpha + (1.0 - alpha) * hhi_norm)
    else:
        privacy = privacy_mean
    privacy = np.clip(privacy, 0.0, 1.0)

    empty = row_sum == 0
    privacy_mean = np.where(empty, np.nan, privacy_mean)
    privacy      = np.where(empty, np.nan, privacy)
    hhi_norm     = np.where(empty, np.nan, hhi_norm)

    return dict(poi_count=row_sum,
                privacy_mean=privacy_mean,
                privacy=privacy,
                hhi_norm=hhi_norm)


# --- 5. Save + visualize ----------------------------------------------------

def save_outputs(result, gs, out_dir: Path, alpha: float, use_conc: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    ny, nx = gs["ny"], gs["nx"]
    as_grid = lambda v: v.reshape(ny, nx)

    np.save(out_dir / "privacy.npy",      as_grid(result["privacy"]))
    np.save(out_dir / "poi_count.npy",    as_grid(result["poi_count"]))

    idx = np.where(result["poi_count"] > 0)[0]
    y = idx // nx
    x = idx % nx
    df = pd.DataFrame({
        "grid_id":      [f"{xi}_{yi}" for xi, yi in zip(x, y)],
        "x":            x,
        "y":            y,
        "lon_center":   gs["min_lon"] + (x + 0.5) * gs["d_lon"],
        "lat_center":   gs["min_lat"] + (y + 0.5) * gs["d_lat"],
        "poi_count":    result["poi_count"][idx].astype(int),
        "privacy_mean": result["privacy_mean"][idx],
        "privacy":      result["privacy"][idx],
        "hhi_norm":     result["hhi_norm"][idx],
    })
    df.to_csv(out_dir / "grid_privacy_scores.csv", index=False)

    meta = {
        "bbox": {k: gs[k] for k in ("min_lon", "max_lon", "min_lat", "max_lat")},
        "nx": nx, "ny": ny,
        "cell_m": gs["cell_m"],
        "d_lon": gs["d_lon"], "d_lat": gs["d_lat"],
        "n_nonempty_cells": int(len(df)),
        "n_total_cells": int(nx * ny),
        "concentration_alpha": alpha,
        "use_concentration_weight": bool(use_conc),
    }
    (out_dir / "grid_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    return df, meta


def plot_two(result, gs, fig_dir: Path, prefix: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    ny, nx = gs["ny"], gs["nx"]
    extent = [gs["min_lon"], gs["max_lon"], gs["min_lat"], gs["max_lat"]]
    cell_m = gs["cell_m"]

    cnt2d = result["poi_count"].reshape(ny, nx)
    cnt2d_log = np.where(cnt2d > 0, np.log1p(cnt2d), np.nan)
    priv2d = result["privacy"].reshape(ny, nx)

    # --- density ---
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(cnt2d_log, origin="lower", extent=extent,
                   cmap="viridis", aspect="auto", interpolation="nearest")
    ax.set_title(f"POI density — {int(cell_m)} m × {int(cell_m)} m cells")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2, linewidth=0.3)
    fig.colorbar(im, ax=ax, shrink=0.85).set_label("log1p(POI count)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_density.png", dpi=160)
    plt.close(fig)

    # --- privacy ---
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(priv2d, origin="lower", extent=extent,
                   cmap="magma", vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    ax.set_title(f"Grid privacy score — {int(cell_m)} m × {int(cell_m)} m cells")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2, linewidth=0.3)
    fig.colorbar(im, ax=ax, shrink=0.85).set_label("privacy (0=public, 1=private)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{prefix}_privacy.png", dpi=160)
    plt.close(fig)


# --- CLI --------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tsv", type=Path, default=DEFAULTS["tsv"],
                   help="POI TSV in loco format (default: %(default)s)")
    p.add_argument("--score-json", type=Path, default=DEFAULTS["score_json"],
                   help="Per-category privacy score JSON (default: %(default)s)")
    p.add_argument("--out-dir", type=Path, default=DEFAULTS["out_dir"],
                   help="Directory for CSV/NPY outputs (default: %(default)s)")
    p.add_argument("--fig-dir", type=Path, default=DEFAULTS["fig_dir"],
                   help="Directory for figures (default: %(default)s)")
    p.add_argument("--fig-prefix", default=DEFAULTS["fig_prefix"],
                   help="Filename prefix for figures (default: %(default)s)")

    p.add_argument("--min-lon", type=float, default=DEFAULTS["min_lon"])
    p.add_argument("--max-lon", type=float, default=DEFAULTS["max_lon"])
    p.add_argument("--min-lat", type=float, default=DEFAULTS["min_lat"])
    p.add_argument("--max-lat", type=float, default=DEFAULTS["max_lat"])
    p.add_argument("--cell-m", type=float, default=DEFAULTS["cell_m"],
                   help="Cell size in metres (default: %(default)s)")

    p.add_argument("--alpha", type=float, default=DEFAULTS["alpha"],
                   help="Concentration weight blend: privacy = mean*(α + (1-α)*hhi_norm). "
                        "α=1 disables concentration weighting. (default: %(default)s)")
    p.add_argument("--no-conc", dest="use_conc", action="store_false",
                   default=DEFAULTS["use_conc"],
                   help="Drop the HHI concentration factor entirely.")
    return p.parse_args()


def main():
    args = parse_args()
    _set_jp_font()

    poi_df = load_pois(args.tsv)
    print(f"POIs loaded:         {len(poi_df)}")

    score_tbl = load_scores(args.score_json)
    cat_ids = sorted(score_tbl.keys())
    cat_scores = np.array([score_tbl[c]["privacy_score"] for c in cat_ids],
                          dtype=np.float32)
    print(f"Categories scored:   {len(cat_ids)}")

    gs = grid_spec(args.min_lon, args.max_lon,
                   args.min_lat, args.max_lat, args.cell_m)
    print(f"Grid:                {gs['nx']} x {gs['ny']}  "
          f"(cell={int(gs['cell_m'])} m, "
          f"d_lon={gs['d_lon']:.6f}°, d_lat={gs['d_lat']:.6f}°)")

    M = build_count_matrix(poi_df, gs, cat_ids)
    print(f"Count matrix:        {M.shape}, nnz={M.nnz}")

    result = compute_privacy(M, cat_scores,
                             alpha=args.alpha, use_conc=args.use_conc)
    df, meta = save_outputs(result, gs, args.out_dir,
                            alpha=args.alpha, use_conc=args.use_conc)
    print(f"Non-empty cells:     {meta['n_nonempty_cells']} / {meta['n_total_cells']}")
    print(f"privacy range:       [{np.nanmin(result['privacy']):.3f}, "
          f"{np.nanmax(result['privacy']):.3f}]")

    plot_two(result, gs, args.fig_dir, args.fig_prefix)
    print(f"\nOutputs:")
    print(f"  {args.out_dir}")
    print(f"  {args.fig_dir}/{args.fig_prefix}_density.png")
    print(f"  {args.fig_dir}/{args.fig_prefix}_privacy.png")


if __name__ == "__main__":
    main()
