"""
Unified evaluation: SSIM (OD avg + occupancy), top-k F1, length JSD,
high-sensitivity OD JSD, high-sensitivity occupancy JSD.

Iterates over Gen_traj_noise_*.pkl files in --gen_dir.
"""
import argparse
import os
import re
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.special import rel_entr
from skimage.metrics import structural_similarity as ssim


# ==========================================================
# 0. CLI arguments
# ==========================================================
def parse_args():
    p = argparse.ArgumentParser(description="Unified trajectory metrics evaluation")
    p.add_argument('--feature_file', type=str,
                   default='data/traj_privacy/nagoya/trajectory_features.npy')
    p.add_argument('--real_traj_file', type=str,
                   default='data/traj_privacy/nagoya/noise_sweep/noise_0.00/traj.npy',
                   help='Real (unperturbed) trajectory .npy, shape [N, T, 2]')
    p.add_argument('--gen_dir', type=str, default='./LDP_result_nagoya',
                   help='Directory containing Gen_traj_noise_*.pkl')
    p.add_argument('--out_csv', type=str, default=None,
                   help='Output CSV path (default: <gen_dir>/metrics_summary.csv)')
    p.add_argument('--grid_size', type=int, default=200)
    p.add_argument('--sigma', type=float, default=1.5)
    p.add_argument('--top_n_pattern', type=int, default=4000,
                   help='K for top-k F1 over most-visited grid cells')
    p.add_argument('--length_bins', type=int, default=50)
    p.add_argument('--high_sens_quantile', type=float, default=0.75,
                   help='Quantile threshold; trajectories with sensitivity '
                        'above this quantile are treated as "high sensitivity"')
    p.add_argument('--od_grid_size', type=int, default=50,
                   help='Grid size used for OD JSD (kept smaller than --grid_size '
                        'to keep the OD support manageable)')
    return p.parse_args()


cli = parse_args()
OUT_CSV = cli.out_csv or os.path.join(cli.gen_dir, "metrics_summary.csv")
os.makedirs(cli.gen_dir, exist_ok=True)

# ==========================================================
# 1. Load real trajectories (privacy-filtered, same as gen scripts)
# ==========================================================
print("Loading real trajectories...")
features = np.load(cli.feature_file)
privacy_budget = features[:, -1]
priv_idx = np.where(privacy_budget > 1e-8)[0]

real_traj_full = np.load(cli.real_traj_file)
real_traj = real_traj_full[priv_idx]
sens_scores = privacy_budget[priv_idx]

# Coordinate range fixed from the full real trajectory pool
all_x = real_traj[:, :, 0].reshape(-1)
all_y = real_traj[:, :, 1].reshape(-1)
X_MIN, X_MAX = float(all_x.min()), float(all_x.max())
Y_MIN, Y_MAX = float(all_y.min()), float(all_y.max())

# High-sensitivity mask (over the privacy-filtered set)
sens_threshold = float(np.quantile(sens_scores, cli.high_sens_quantile))
high_sens_mask = sens_scores >= sens_threshold

print(f"Real traj count (privacy-filtered): {len(real_traj)}")
print(f"High-sensitivity threshold (q={cli.high_sens_quantile}): {sens_threshold:.4f}")
print(f"High-sensitivity count: {int(high_sens_mask.sum())}")


# ==========================================================
# 2. Helpers
# ==========================================================
def jsd(p, q, eps=1e-12):
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    return 0.5 * np.sum(rel_entr(p, m)) + 0.5 * np.sum(rel_entr(q, m))


def coord_to_grid(points, grid_size):
    nx = (points[:, 0] - X_MIN) / (X_MAX - X_MIN + 1e-12)
    ny = (points[:, 1] - Y_MIN) / (Y_MAX - Y_MIN + 1e-12)
    gx = np.clip((nx * grid_size).astype(np.int64), 0, grid_size - 1)
    gy = np.clip((ny * grid_size).astype(np.int64), 0, grid_size - 1)
    return gx, gy


# ---------- SSIM building blocks ----------
def density_heatmap(traj, mode):
    """mode in {'start', 'end', 'all'}; returns gaussian-smoothed, max-normalized 2D map."""
    heat = np.zeros((cli.grid_size, cli.grid_size), dtype=np.float32)
    if mode == "start":
        pts = traj[:, 0, :]
    elif mode == "end":
        pts = traj[:, -1, :]
    elif mode == "all":
        pts = traj.reshape(-1, 2)
    else:
        raise ValueError(mode)
    gx, gy = coord_to_grid(pts, cli.grid_size)
    np.add.at(heat, (gy, gx), 1)
    if cli.sigma > 0:
        heat = gaussian_filter(heat, sigma=cli.sigma)
    heat = heat / (heat.max() + 1e-12)
    return heat


def ssim_pair(real, gen, mode):
    return ssim(density_heatmap(real, mode), density_heatmap(gen, mode), data_range=1.0)


# ---------- OD / occupancy JSD (sparse) ----------
def od_distribution(traj, grid_size):
    """Sparse OD: returns (keys, counts) where key encodes (sx, sy, ex, ey)."""
    sx, sy = coord_to_grid(traj[:, 0, :], grid_size)
    ex, ey = coord_to_grid(traj[:, -1, :], grid_size)
    keys = (sx * grid_size ** 3 + sy * grid_size ** 2 +
            ex * grid_size + ey).astype(np.int64)
    u, c = np.unique(keys, return_counts=True)
    return u, c.astype(np.float64)


def occupancy_distribution(traj, grid_size):
    """Sparse occupancy: returns (keys, counts) over visited grid cells (all points)."""
    pts = traj.reshape(-1, 2)
    gx, gy = coord_to_grid(pts, grid_size)
    keys = (gy * grid_size + gx).astype(np.int64)
    u, c = np.unique(keys, return_counts=True)
    return u, c.astype(np.float64)


def sparse_jsd(keys_r, vals_r, keys_g, vals_g):
    all_keys = np.union1d(keys_r, keys_g)
    pr = np.zeros(len(all_keys), dtype=np.float64)
    pg = np.zeros(len(all_keys), dtype=np.float64)
    pr[np.searchsorted(all_keys, keys_r)] = vals_r
    pg[np.searchsorted(all_keys, keys_g)] = vals_g
    return jsd(pr, pg)


# ---------- Length JSD ----------
def trajectory_lengths(traj):
    diff = np.diff(traj, axis=1)
    seg = np.linalg.norm(diff, axis=2)
    return seg.sum(axis=1)


def length_jsd(real, gen):
    r = trajectory_lengths(real)
    g = trajectory_lengths(gen)
    hr, bins = np.histogram(r, bins=cli.length_bins, density=True)
    hg, _ = np.histogram(g, bins=bins, density=True)
    return jsd(hr, hg)


# ---------- Top-K F1 ----------
def topk_f1(real, gen):
    def topk_cells(traj):
        pts = traj.reshape(-1, 2)
        gx, gy = coord_to_grid(pts, cli.grid_size)
        idx = gy * cli.grid_size + gx
        u, c = np.unique(idx, return_counts=True)
        order = np.argsort(-c)[:cli.top_n_pattern]
        return set(u[order].tolist())

    P = topk_cells(real)
    Pg = topk_cells(gen)
    tp = len(P & Pg)
    precision = tp / (len(Pg) + 1e-12)
    recall = tp / (len(P) + 1e-12)
    return 2 * precision * recall / (precision + recall + 1e-12)


# ==========================================================
# 3. Discover trajectories (supports two layouts)
#    A) <gen_dir>/Gen_traj_noise_<X.XX>.pkl                (DiffTraj outputs)
#    B) <gen_dir>/noise_<X.XX>/traj.npy                    (raw noise_sweep)
# ==========================================================
PKL_PAT = re.compile(r"Gen_traj_noise_([0-9.]+)\.pkl$")
DIR_PAT = re.compile(r"noise_([0-9.]+)$")


def discover_traj_sources(gen_dir):
    items = []  # list of (noise_level: float, loader: callable -> np.ndarray)
    for entry in sorted(os.listdir(gen_dir)):
        full = os.path.join(gen_dir, entry)
        m_pkl = PKL_PAT.match(entry)
        m_dir = DIR_PAT.match(entry)
        if m_pkl and os.path.isfile(full):
            noise = float(m_pkl.group(1))
            items.append((noise, "pkl", full))
        elif m_dir and os.path.isdir(full):
            traj_npy = os.path.join(full, "traj.npy")
            if os.path.isfile(traj_npy):
                noise = float(m_dir.group(1))
                items.append((noise, "npy", traj_npy))
    return items


def load_traj(kind, path):
    if kind == "pkl":
        with open(path, "rb") as f:
            return np.array(pickle.load(f))
    elif kind == "npy":
        return np.load(path)
    raise ValueError(kind)


sources = discover_traj_sources(cli.gen_dir)
print(f"\nFound {len(sources)} trajectory source(s) in {cli.gen_dir}")
records = []

for noise, kind, path in tqdm(sources):
    gen_traj = load_traj(kind, path)

    # If this dir IS the noise_sweep itself, traj.npy is unfiltered (length matches features).
    # Apply the same privacy filter so it lines up with real_traj / sens_scores.
    if kind == "npy" and len(gen_traj) == len(features):
        gen_traj = gen_traj[priv_idx]

    n = min(len(real_traj), len(gen_traj))
    real_sub = real_traj[:n]
    gen_sub = gen_traj[:n]
    hs_mask = high_sens_mask[:n]

    real_hs = real_sub[hs_mask]
    gen_hs = gen_sub[hs_mask]

    # SSIM (OD average + occupancy)
    s_start = ssim_pair(real_sub, gen_sub, "start")
    s_end = ssim_pair(real_sub, gen_sub, "end")
    s_od_avg = 0.5 * (s_start + s_end)
    s_occ = ssim_pair(real_sub, gen_sub, "all")

    # Top-K F1
    f1 = topk_f1(real_sub, gen_sub)

    # Length JSD
    l_jsd = length_jsd(real_sub, gen_sub)

    # High-sensitivity OD / occupancy JSD (sparse)
    if len(real_hs) > 0 and len(gen_hs) > 0:
        kr, vr = od_distribution(real_hs, cli.od_grid_size)
        kg, vg = od_distribution(gen_hs, cli.od_grid_size)
        od_jsd_hs = sparse_jsd(kr, vr, kg, vg)

        kr, vr = occupancy_distribution(real_hs, cli.grid_size)
        kg, vg = occupancy_distribution(gen_hs, cli.grid_size)
        occ_jsd_hs = sparse_jsd(kr, vr, kg, vg)
    else:
        od_jsd_hs = float("nan")
        occ_jsd_hs = float("nan")

    records.append({
        "noise": noise,
        "ssim_od_avg": s_od_avg,
        "ssim_start": s_start,
        "ssim_end": s_end,
        "ssim_occupancy": s_occ,
        "topk_f1": f1,
        "length_jsd": l_jsd,
        "od_jsd_high_sens": od_jsd_hs,
        "occupancy_jsd_high_sens": occ_jsd_hs,
    })

# ==========================================================
# 4. Save
# ==========================================================
df = pd.DataFrame(records).sort_values("noise").reset_index(drop=True)

print("\n========== Evaluation Summary ==========")
with pd.option_context('display.max_columns', None, 'display.width', 200):
    print(df)

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved to: {OUT_CSV}")
