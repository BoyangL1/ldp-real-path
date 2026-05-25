"""Trajectory Privacy Dataset (compact input).

Converted from `3-traj-privacy-compact.ipynb` for server runs.
Figures are saved under `figs/` instead of shown interactively.

Run with defaults (Nagoya):
    python 3-traj-privacy-compact.py

Override any input / output / hyperparameter from the CLI — see `--help`.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter
from scipy.stats import linregress, norm
from skimage.metrics import structural_similarity as ssim

# ---- Defaults (overridden by CLI in main()) ----
COMPACT_CSV  = Path('data/sim_nagoya_20230426_traj_compact.csv.gz')
GRID_META    = Path('data/privacy_outputs/nagoya/grid_meta.json')
PRIVACY_CSV  = Path('data/privacy_outputs/nagoya/grid_privacy_scores.csv')
OUTPUT_ROOT  = Path('data/traj_privacy/nagoya')
FIG_DIR      = Path('figs')
CITY         = 'nagoya'

GRID_SIZE_KM     = 0.1
TARGET_LEN       = 20
NOISE_LEVELS     = np.linspace(0.0, 1.0, num=11)
MAX_SPEED_KMH    = 100
MAX_DRAW         = 1000
GRID_HEATMAP     = 200
SIGMA            = 1.5
SAMPLE_SIZE      = 2000
RANDOM_SEED      = 42

rcParams['font.family'] = 'serif'
rcParams['font.serif']  = ['Times New Roman', 'Times', 'STIXGeneral']
rcParams['mathtext.fontset'] = 'stix'


@dataclass
class DiffusionConfig:
    target_len: int = 20
    beta_start: float = 1e-4
    beta_end:   float = 0.01
    num_steps:  int   = 1000
    delta_f0:   float = 5.0
    delta:      float = 1e-6


CFG = DiffusionConfig()

PATHS: dict = {}


def build_paths(output_root: Path) -> dict:
    return {
        'aggregated'    : output_root / 'aggregated_trajectories.parquet',
        'enriched'      : output_root / 'aggregated_trajectories_with_privacy.parquet',
        'features'      : output_root / 'trajectory_features.npy',
        'feature_stats' : output_root / 'trajectory_feature_stats.csv',
        'norm_stats'    : output_root / 'trajectory_tensor_stats.npz',
        'noise_dir'     : output_root / 'noise_sweep',
        'summary'       : output_root / 'summary.csv',
    }


# ---- Helpers ----
def load_grid_meta(path: Path = None) -> dict:
    return json.loads(Path(path or GRID_META).read_text())


def latlon_to_grid(lat: np.ndarray, lon: np.ndarray, meta: dict):
    bb = meta['bbox']
    x = np.floor((lon - bb['min_lon']) / meta['d_lon']).astype(np.int64)
    y = np.floor((lat - bb['min_lat']) / meta['d_lat']).astype(np.int64)
    x = np.clip(x, 0, meta['nx'] - 1)
    y = np.clip(y, 0, meta['ny'] - 1)
    return x, y


def load_compact(path: Path = None, meta: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or COMPACT_CSV)
    meta = meta or load_grid_meta()
    x, y = latlon_to_grid(df['lat'].to_numpy(), df['lon'].to_numpy(), meta)
    df['x'] = x.astype(np.int32)
    df['y'] = y.astype(np.int32)
    df = df.sort_values(['ID1', 'date', 'traj_id', 'unixtime'], kind='mergesort').reset_index(drop=True)
    return df


def aggregate_trajectories(compact_df: pd.DataFrame) -> pd.DataFrame:
    tj_key = ['ID1', 'date', 'traj_id']
    df = compact_df.copy()

    dx = df.groupby(tj_key)['x'].diff()
    dy = df.groupby(tj_key)['y'].diff()
    df['step_units'] = np.sqrt(dx ** 2 + dy ** 2).fillna(0.0)

    grp = df.groupby(tj_key, sort=False)
    traj = grp.agg(
        traj_num         = ('x', 'size'),
        min_ut           = ('unixtime', 'min'),
        max_ut           = ('end_unixtime', 'max'),
        total_dist_units = ('step_units', 'sum'),
    )
    traj_lists = grp[['x', 'y']].apply(
        lambda g: list(zip(g['x'].astype(int), g['y'].astype(int)))
    ).rename('trajectory_list')
    traj = traj.join(traj_lists).reset_index()

    day_start = (traj['min_ut'] // 86400) * 86400
    traj['departure_bucket'] = (((traj['min_ut'] - day_start) // 1800) % 48).astype(np.int32)
    traj['departure_time']   = (traj['min_ut'] - day_start) / 3600.0
    traj['trip_time']        = (traj['max_ut'] - traj['min_ut']) / 3600.0
    traj['trip_distance']    = traj['total_dist_units'] * GRID_SIZE_KM

    traj['avg_dis'] = 0.0
    m = traj['traj_num'] > 1
    traj.loc[m, 'avg_dis'] = traj.loc[m, 'trip_distance'] / (traj.loc[m, 'traj_num'] - 1)

    traj['avg_speed'] = 0.0
    m = traj['trip_time'] > 0
    traj.loc[m, 'avg_speed'] = traj.loc[m, 'trip_distance'] / traj.loc[m, 'trip_time']

    traj['trip_length'] = traj['trajectory_list'].apply(
        lambda t: float(np.linalg.norm(np.array(t[-1]) - np.array(t[0])) * GRID_SIZE_KM) if len(t) > 0 else 0.0
    )

    traj = traj.drop(columns=['min_ut', 'max_ut', 'total_dist_units'])
    traj = traj[traj['avg_speed'] < MAX_SPEED_KMH].reset_index(drop=True)
    traj['global_seq_id'] = np.arange(len(traj), dtype=np.int64)
    return traj[[
        'global_seq_id', 'ID1', 'date', 'traj_id', 'trajectory_list', 'traj_num',
        'departure_bucket', 'departure_time', 'trip_distance', 'trip_time',
        'trip_length', 'avg_dis', 'avg_speed',
    ]]


def attach_privacy_and_build_features(traj_df: pd.DataFrame, privacy_df: pd.DataFrame):
    df = traj_df.copy()
    privacy_map = dict(zip(privacy_df['grid_id'], privacy_df['privacy_mean']))

    df['start_point'] = df['trajectory_list'].apply(lambda t: tuple(t[0])  if len(t) > 0 else (np.nan, np.nan))
    df['end_point']   = df['trajectory_list'].apply(lambda t: tuple(t[-1]) if len(t) > 0 else (np.nan, np.nan))
    df['start_xy']    = df['start_point'].apply(lambda p: f'{int(p[0])}_{int(p[1])}')
    df['end_xy']      = df['end_point'].apply(lambda p: f'{int(p[0])}_{int(p[1])}')

    df['start_privacy']       = df['start_xy'].map(privacy_map).fillna(0.0)
    df['end_privacy']         = df['end_xy'].map(privacy_map).fillna(0.0)
    df['privacy_budget_base'] = (df['start_privacy'] + df['end_privacy']) / 2.0

    if len(df) > 1:
        ranks = pd.Series(df['privacy_budget_base'].values).rank(method='average').values
        p_trigger = (ranks - 1) / (len(ranks) - 1)
    else:
        p_trigger = np.zeros(len(df), dtype=float)
    df['p_trigger'] = p_trigger

    rng = np.random.default_rng(RANDOM_SEED)
    trigger = rng.binomial(1, np.clip(df['p_trigger'].to_numpy(), 0.0, 1.0))
    strength = rng.beta(2, 2, size=len(df))
    df['privacy_budget'] = np.clip(trigger * strength, 0.0, 1.0)

    all_points = pd.concat([df['start_point'], df['end_point']], axis=0)
    ids, _ = pd.factorize(all_points)
    df['start_id_group'] = ids[:len(df)]
    df['end_id_group']   = ids[len(df):]

    cont_cols = ['trip_distance', 'trip_time', 'trip_length', 'avg_dis', 'avg_speed']
    id_cols   = ['start_id_group', 'end_id_group']
    departure = df[['departure_bucket']].to_numpy(dtype=np.float32)

    means = df[cont_cols].mean()
    stds  = df[cont_cols].std().replace(0, 1.0).fillna(1.0)
    cont_norm = ((df[cont_cols] - means) / stds).to_numpy(dtype=np.float32)
    ids_np    = df[id_cols].to_numpy(dtype=np.float32)
    privacy   = df[['privacy_budget']].to_numpy(dtype=np.float32)

    features = np.concatenate([departure, cont_norm, ids_np, privacy], axis=1).astype(np.float32)
    feature_stats = pd.DataFrame({'feature': cont_cols, 'mean': means.values, 'std': stds.values})
    return df, features, feature_stats


def resample_trajectory(traj, target_len: int) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)
    if len(traj) == 0:
        return np.zeros((target_len, 2), dtype=np.float32)
    if len(traj) == target_len:
        return traj.astype(np.float32)
    t_old = np.linspace(0, 1, len(traj))
    t_new = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, traj.shape[1]), dtype=np.float32)
    for d in range(traj.shape[1]):
        out[:, d] = np.interp(t_new, t_old, traj[:, d])
    return out


def trajectories_to_tensor(traj_df: pd.DataFrame, target_len: int) -> np.ndarray:
    processed = [resample_trajectory(t, target_len) for t in traj_df['trajectory_list'].values]
    return np.stack(processed).astype(np.float32)


def normalize_tensor_zscore(tensor: np.ndarray, eps: float = 1e-8):
    mean = tensor.mean(axis=(0, 1), keepdims=True)
    std  = tensor.std(axis=(0, 1), keepdims=True)
    return ((tensor - mean) / (std + eps)).astype(np.float32), {
        'mean': mean.astype(np.float32), 'std': std.astype(np.float32),
    }


def compute_alpha_bar(beta_start, beta_end, num_steps):
    betas = np.linspace(beta_start, beta_end, num_steps)
    return np.cumprod(1 - betas)


def eps_per_timestep(alpha_bar, delta_f0=5.0, delta=1e-6):
    alpha_bar = np.clip(alpha_bar, 1e-12, 1 - 1e-12)
    a_t = np.sqrt(alpha_bar) * delta_f0 / np.sqrt(1 - alpha_bar)
    return 0.5 * a_t ** 2 - a_t * norm.ppf(delta)


def epsilon_to_timestep(epsilon, eps_t):
    idx = np.where(eps_t <= float(epsilon))[0]
    return int(idx[0]) if len(idx) else len(eps_t) - 1


def budget_to_epsilon(p, eps_min, eps_max):
    p = np.clip(p, 0.0, 1.0)
    log_eps = (1 - p) * np.log(eps_max) + p * np.log(eps_min)
    return float(np.exp(log_eps))


def diffusion_forward_by_timestep(x0, timesteps, alpha_bar):
    n, L, D = x0.shape
    eps = np.random.randn(n, L, D).astype(np.float32)
    out = np.zeros_like(x0, dtype=np.float32)
    for i in range(n):
        a = alpha_bar[int(timesteps[i])]
        out[i] = np.sqrt(a) * x0[i] + np.sqrt(1 - a) * eps[i]
    return out


def generate_density_heatmap(traj_array, grid_size, mode='start', sigma=1.5, bounds=None):
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    pts = traj_array[:, 0, :] if mode == 'start' else traj_array[:, -1, :]
    if bounds is None:
        x_min, x_max = traj_array[..., 0].min(), traj_array[..., 0].max()
        y_min, y_max = traj_array[..., 1].min(), traj_array[..., 1].max()
    else:
        x_min, x_max, y_min, y_max = bounds
    nx = (pts[:, 0] - x_min) / (x_max - x_min + 1e-12)
    ny = (pts[:, 1] - y_min) / (y_max - y_min + 1e-12)
    gx = np.clip((nx * grid_size).astype(int), 0, grid_size - 1)
    gy = np.clip((ny * grid_size).astype(int), 0, grid_size - 1)
    np.add.at(heatmap, (gy, gx), 1)
    if sigma > 0:
        heatmap = gaussian_filter(heatmap, sigma=sigma)
    return heatmap


def compute_normalized_dtw(t1, t2):
    n = min(len(t1), len(t2))
    if n == 0:
        return 0.0
    return float(np.linalg.norm(t1[:n] - t2[:n], axis=1).mean())


# ---- Build the dataset ----
def process_city(cfg: DiffusionConfig = CFG) -> dict:
    meta        = load_grid_meta()
    compact_df  = load_compact(meta=meta)
    privacy_df  = pd.read_csv(PRIVACY_CSV)

    traj_df = aggregate_trajectories(compact_df)
    traj_df.to_parquet(PATHS['aggregated'], index=False)

    enriched, features, feat_stats = attach_privacy_and_build_features(traj_df, privacy_df)
    enriched.to_parquet(PATHS['enriched'], index=False)
    np.save(PATHS['features'], features)
    feat_stats.to_csv(PATHS['feature_stats'], index=False)

    traj_tensor = trajectories_to_tensor(enriched, cfg.target_len)
    traj_tensor, norm_stats = normalize_tensor_zscore(traj_tensor)
    np.savez(PATHS['norm_stats'], **norm_stats)

    alpha_bar = compute_alpha_bar(cfg.beta_start, cfg.beta_end, cfg.num_steps)
    eps_t     = eps_per_timestep(alpha_bar, delta_f0=cfg.delta_f0, delta=cfg.delta)
    eps_min, eps_max = eps_t[-1], eps_t[0]

    PATHS['noise_dir'].mkdir(parents=True, exist_ok=True)
    for nl in NOISE_LEVELS:
        p_scaled = np.clip(features[:, -1] * nl, 0.0, 1.0)
        epsilon  = np.array([budget_to_epsilon(p, eps_min, eps_max) for p in p_scaled])
        t_scaled = np.array([epsilon_to_timestep(e, eps_t) for e in epsilon], dtype=np.int32)
        traj_noisy = diffusion_forward_by_timestep(traj_tensor, t_scaled, alpha_bar)

        sub = PATHS['noise_dir'] / f'noise_{nl:.2f}'
        sub.mkdir(parents=True, exist_ok=True)
        np.save(sub / 'traj.npy', traj_noisy)
        feats_nl = features.copy()
        feats_nl[:, -1] = t_scaled
        np.save(sub / 'traj_features.npy', feats_nl)

    summary = {
        'city'             : CITY,
        'num_compact_rows' : int(len(compact_df)),
        'num_trajectories' : int(len(enriched)),
        'num_privacy_traj' : int((enriched['privacy_budget'] > 1e-8).sum()),
        'privacy_share'    : float((enriched['privacy_budget'] > 1e-8).mean()),
        'feature_dim'      : int(features.shape[1]),
        'tensor_shape'     : str(tuple(traj_tensor.shape)),
        'noise_levels'     : int(len(NOISE_LEVELS)),
    }
    pd.DataFrame([summary]).to_csv(PATHS['summary'], index=False)
    return summary


def inspect_outputs():
    print('output root:', OUTPUT_ROOT)
    for p in sorted(OUTPUT_ROOT.rglob('*')):
        rel = p.relative_to(OUTPUT_ROOT)
        if p.is_file():
            print(f'  {rel}  ({p.stat().st_size / 1e6:.2f} MB)')
        else:
            print(f'  {rel}/')

    enriched = pd.read_parquet(PATHS['enriched'])
    print('\nenriched columns:', list(enriched.columns))
    print(enriched[[
        'ID1', 'traj_id', 'traj_num', 'trip_distance', 'trip_time', 'avg_speed',
        'start_xy', 'end_xy', 'privacy_budget_base', 'privacy_budget',
    ]].head(10).to_string())


def plot_noise_sweep():
    all_trajs, xs, ys = {}, [], []
    for nl in NOISE_LEVELS:
        t = np.load(PATHS['noise_dir'] / f'noise_{nl:.2f}' / 'traj.npy')
        if len(t) > MAX_DRAW:
            t = t[np.random.choice(len(t), MAX_DRAW, replace=False)]
        all_trajs[float(nl)] = t
        xs.append(t[..., 0].ravel())
        ys.append(t[..., 1].ravel())

    xs = np.concatenate(xs); ys = np.concatenate(ys)
    margin_x = (xs.max() - xs.min()) * 0.02
    margin_y = (ys.max() - ys.min()) * 0.02
    xlim = (xs.min() - margin_x, xs.max() + margin_x)
    ylim = (ys.min() - margin_y, ys.max() + margin_y)

    n_panels = len(NOISE_LEVELS)
    n_cols = min(6, n_panels)
    n_rows = int(np.ceil((n_panels + 1) / n_cols))  # +1 for blank panel like original
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(4 * n_cols, 10), 4 * n_rows), dpi=150)
    axes = np.atleast_1d(axes).flatten()
    for ax, nl in zip(axes, NOISE_LEVELS):
        t = all_trajs[float(nl)]
        lc = LineCollection(t, colors='#444', linewidths=0.3, alpha=0.3, rasterized=True)
        ax.add_collection(lc)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'noise = {nl:.1f}', fontsize=16)
    for ax in axes[len(NOISE_LEVELS):]:
        ax.axis('off')
    fig.suptitle(f'Trajectory Privacy Trade-off: {CITY.capitalize()} (compact)', fontsize=20)
    plt.tight_layout()
    out = FIG_DIR / f'3_{CITY}_noise_sweep.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')


def compute_ssim():
    traj_real = np.load(PATHS['noise_dir'] / 'noise_0.00' / 'traj.npy')
    features  = np.load(PATHS['features'])
    priv_idx  = np.where(features[:, -1] > 1e-8)[0]
    traj_real_priv = traj_real[priv_idx]
    ref_bounds = (
        traj_real_priv[..., 0].min(), traj_real_priv[..., 0].max(),
        traj_real_priv[..., 1].min(), traj_real_priv[..., 1].max(),
    )

    img_real = generate_density_heatmap(
        traj_real_priv, GRID_HEATMAP, mode='start', sigma=SIGMA, bounds=ref_bounds,
    )
    base_max = img_real.max() + 1e-12

    ssim_rows = []
    for nl in NOISE_LEVELS:
        t = np.load(PATHS['noise_dir'] / f'noise_{nl:.2f}' / 'traj.npy')[priv_idx]
        img = generate_density_heatmap(t, GRID_HEATMAP, mode='start', sigma=SIGMA, bounds=ref_bounds)
        s = ssim(img_real / base_max, img / base_max, data_range=1.0)
        ssim_rows.append({'noise': float(nl), 'mode': 'start', 'ssim': float(s)})

    ssim_df = pd.DataFrame(ssim_rows)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ssim_df.plot(x='noise', y='ssim', marker='o', legend=False, ax=ax,
                 title='Start-point density SSIM vs noise level')
    ax.set_ylabel('SSIM')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / f'3_{CITY}_ssim_vs_noise.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')
    ssim_df.to_csv(OUTPUT_ROOT / 'ssim_vs_noise.csv', index=False)
    print(ssim_df.to_string())
    return ssim_df, traj_real, features, priv_idx


def compute_dtw(traj_real, features, priv_idx):
    traj_real_priv = traj_real[priv_idx]
    privacy_priv   = features[priv_idx, -1]

    rng = np.random.default_rng(RANDOM_SEED)
    dtw_rows = []
    for nl in NOISE_LEVELS:
        t_noisy = np.load(PATHS['noise_dir'] / f'noise_{nl:.2f}' / 'traj.npy')[priv_idx]
        n = min(SAMPLE_SIZE, len(traj_real_priv))
        sel = rng.choice(len(traj_real_priv), n, replace=False)

        errs, ps = [], []
        for i in sel:
            e = compute_normalized_dtw(traj_real_priv[i], t_noisy[i])
            if np.isfinite(e):
                errs.append(e); ps.append(privacy_priv[i])
        errs = np.asarray(errs); ps = np.asarray(ps)

        if nl == 0.0 or np.std(ps) < 1e-6 or len(ps) < 10:
            dtw_rows.append({
                'noise': float(nl), 'slope': 0.0, 'pearson_r': 0.0,
                'r2': 1.0 if nl == 0.0 else np.nan,
                'dtw_mean': float(np.log(errs + 1e-6).mean()) if len(errs) else 0.0,
                'intercept': np.nan,
            })
            continue

        y = np.log(errs + 1e-6)
        slope, intercept, r, _, _ = linregress(ps, y)
        dtw_rows.append({
            'noise': float(nl), 'slope': float(slope), 'pearson_r': float(r),
            'r2': float(r ** 2), 'dtw_mean': float(y.mean()),
            'intercept': float(intercept),
        })

    dtw_df = pd.DataFrame(dtw_rows)
    dtw_df.to_csv(OUTPUT_ROOT / 'dtw_vs_noise.csv', index=False)
    print(dtw_df.to_string())
    return dtw_df


def plot_tradeoff(dtw_df, ssim_df):
    tradeoff = dtw_df.merge(ssim_df[['noise', 'ssim']], on='noise', how='left').sort_values('slope').reset_index(drop=True)
    tradeoff.loc[tradeoff['noise'] == 0.0, 'slope'] = 0.0

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(tradeoff['slope'], tradeoff['ssim'], s=90, alpha=0.85, edgecolors='k')
    ax.plot(tradeoff['slope'], tradeoff['ssim'], linestyle='--', alpha=0.6)
    for _, row in tradeoff.iterrows():
        ax.text(row['slope'] + 0.05, row['ssim'] + 0.003, f"{row['noise']:.2f}", fontsize=9, alpha=0.8)
    ax.set_xlabel('Privacy -> utility sensitivity (slope)')
    ax.set_ylabel('Spatial SSIM')
    ax.set_title(f'Privacy sensitivity vs spatial similarity ({CITY.capitalize()})')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / f'3_{CITY}_tradeoff.png'
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out}')
    tradeoff.to_csv(OUTPUT_ROOT / 'tradeoff.csv', index=False)
    print(tradeoff.to_string())


def parse_args():
    p = argparse.ArgumentParser(
        description='Build trajectory privacy dataset + noise sweep + analysis plots.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # I/O
    p.add_argument('--compact-csv', type=Path, default=COMPACT_CSV,
                   help='Compact trajectory CSV produced by 1-split-traj.py')
    p.add_argument('--grid-meta', type=Path, default=GRID_META,
                   help='grid_meta.json from 0-grid_privacy.py')
    p.add_argument('--privacy-csv', type=Path, default=PRIVACY_CSV,
                   help='grid_privacy_scores.csv from 0-grid_privacy.py')
    p.add_argument('--output-root', type=Path, default=None,
                   help='Output directory (default: data/traj_privacy/<city>)')
    p.add_argument('--fig-dir', type=Path, default=FIG_DIR,
                   help='Where to save PNG figures')
    p.add_argument('--city', type=str, default=CITY,
                   help='City name used in titles and summary')
    # Hyperparameters
    p.add_argument('--target-len', type=int, default=TARGET_LEN,
                   help='Resampled trajectory length')
    p.add_argument('--noise-min', type=float, default=0.0)
    p.add_argument('--noise-max', type=float, default=1.0)
    p.add_argument('--noise-count', type=int, default=11,
                   help='Number of noise levels (inclusive of endpoints)')
    p.add_argument('--sample-size', type=int, default=SAMPLE_SIZE,
                   help='Trajectories sampled per noise level for DTW regression')
    p.add_argument('--max-draw', type=int, default=MAX_DRAW,
                   help='Max trajectories drawn per noise panel')
    p.add_argument('--grid-heatmap', type=int, default=GRID_HEATMAP,
                   help='Heatmap grid size for SSIM')
    p.add_argument('--sigma', type=float, default=SIGMA,
                   help='Gaussian blur sigma for heatmap')
    p.add_argument('--grid-size-km', type=float, default=GRID_SIZE_KM,
                   help='Cell size in km (must match grid_meta.json)')
    p.add_argument('--max-speed-kmh', type=float, default=MAX_SPEED_KMH,
                   help='Drop trajectories with avg speed above this cutoff')
    p.add_argument('--seed', type=int, default=RANDOM_SEED)
    # Flow control
    p.add_argument('--skip-process', action='store_true',
                   help='Skip data build and go straight to plots (noise_sweep/ must exist)')
    return p.parse_args()


def main():
    global COMPACT_CSV, GRID_META, PRIVACY_CSV, OUTPUT_ROOT, FIG_DIR, CITY
    global GRID_SIZE_KM, TARGET_LEN, NOISE_LEVELS, MAX_SPEED_KMH
    global MAX_DRAW, GRID_HEATMAP, SIGMA, SAMPLE_SIZE, RANDOM_SEED
    global CFG, PATHS

    args = parse_args()

    COMPACT_CSV   = args.compact_csv
    GRID_META     = args.grid_meta
    PRIVACY_CSV   = args.privacy_csv
    CITY          = args.city
    OUTPUT_ROOT   = args.output_root or Path(f'data/traj_privacy/{CITY}')
    FIG_DIR       = args.fig_dir
    GRID_SIZE_KM  = args.grid_size_km
    TARGET_LEN    = args.target_len
    NOISE_LEVELS  = np.linspace(args.noise_min, args.noise_max, num=args.noise_count)
    MAX_SPEED_KMH = args.max_speed_kmh
    MAX_DRAW      = args.max_draw
    GRID_HEATMAP  = args.grid_heatmap
    SIGMA         = args.sigma
    SAMPLE_SIZE   = args.sample_size
    RANDOM_SEED   = args.seed

    CFG   = DiffusionConfig(target_len=TARGET_LEN)
    PATHS = build_paths(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    print(f'city={CITY} output_root={OUTPUT_ROOT} fig_dir={FIG_DIR}')
    print(f'noise_levels={list(np.round(NOISE_LEVELS, 3))}')

    if not args.skip_process:
        summary = process_city()
        print('summary:', summary)
        inspect_outputs()
    else:
        assert PATHS['noise_dir'].exists(), f'--skip-process set but {PATHS["noise_dir"]} does not exist'
        print('skip-process: reusing existing noise_sweep outputs')

    plot_noise_sweep()
    ssim_df, traj_real, features, priv_idx = compute_ssim()
    dtw_df = compute_dtw(traj_real, features, priv_idx)
    plot_tradeoff(dtw_df, ssim_df)


if __name__ == '__main__':
    main()
