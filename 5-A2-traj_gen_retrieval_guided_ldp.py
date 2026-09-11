"""Server-ready retrieval-guided generation: one scalar weight per trajectory.

Default: adaptive SNR k=0.1, gamma=1, lambda_max=1, categorical OD-ID cost=4.
The weight is uniform across all points and constant over guided denoising steps.
Guidance is active only for t <= t_i. Use --no-adaptive for a fixed-weight control.
See README.md and run_adaptive_4cities.sh for generation and evaluation commands.
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
import pickle
import json
from scipy.stats import rankdata

from utils.Traj_UNet import *
from utils.config import args
from utils.utils import *


# =========================
# 0. CLI arguments
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Retrieval-guided DDIM generation for LDP-DiffTraj")
    # data / model
    p.add_argument('--traj_path', type=str,
                   default='data/traj_privacy/nagoya/noise_sweep/noise_0.00/traj.npy',
                   help='Clean reference trajectories aligned with --head_path rows ([N,Len,2] or [N,2,Len])')
    p.add_argument('--head_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_features.npy',
                   help='Global head: last col = privacy score (defines clean/private split)')
    p.add_argument('--train_head_dir', type=str,
                   default='data/traj_privacy/nagoya/noise_sweep',
                   help='Dir with per-noise training heads noise_{nl}/traj_features.npy whose '
                        'last col = diffusion timestep t_i (matches training). Empty/missing => fallback.')
    p.add_argument('--root', type=str, default='./LDP-DiffTraj_nagoya')
    p.add_argument('--result_root', type=str, default='./LDP_result_nagoya_guided')
    p.add_argument('--run_tag', type=str, default='',
                   help='Sub-folder name under --result_root for this run. '
                        'Empty => auto-named from guidance hyper-parameters.')
    p.add_argument('--noise_prefix', type=str, default='nagoya_noise_')
    p.add_argument('--noise_levels', type=str, default='',
                   help='Comma-separated subset, e.g. "0.00,0.10". Empty = all.')
    p.add_argument('--ckpt_name', type=str, default='unet_1000.pt')
    # sampling
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--timesteps', type=int, default=100)
    p.add_argument('--eta', type=float, default=0.0)
    p.add_argument('--cuda_device', type=str, default='0')
    # retrieval guidance
    p.add_argument('--top_m', type=int, default=500, help='Stage-1 coarse candidates per sample')
    p.add_argument('--top_k', type=int, default=20, help='Stage-2 fine neighbours used for guidance')
    p.add_argument('--guide_lambdas', type=str, default='1.0',
                   help='Comma-separated guidance levels to sweep. Each level is generated '
                        'independently and saved to its own sub-folder guide_<level>/. '
                        'A single value (e.g. "0.1") runs just that level.')
    p.add_argument('--guide_lambda', type=float, default=0.1,
                   help='[deprecated] single mixing weight; used only when --guide_lambdas is empty.')
    # adaptive (per-trajectory) guidance strength
    p.add_argument('--adaptive', action=argparse.BooleanOptionalAction, default=True,
                   help='Per-trajectory adaptive guidance: noisier trajectories (larger t_i) '
                        'are guided more strongly, cleaner ones more weakly. Each value in '
                        '--guide_lambdas is then the PEAK strength lam_max.')
    p.add_argument('--lam_ratio_min', type=float, default=0.0,
                   help='[adaptive] lam_min = lam_ratio_min * lam_max, i.e. the strength given '
                        'to the cleanest trajectories. 0 => they get no guidance at all.')
    p.add_argument('--adapt_gamma', type=float, default=1.0,
                   help='[adaptive] lambda_i = lam_min + (lam_max-lam_min)*s_i**adapt_gamma. '
                        '>1 reserves strong guidance for the noisiest tail, <1 spreads it out.')
    p.add_argument('--adapt_key', type=str, default='ti', choices=['ti', 'privacy'],
                   help='[adaptive] noisiness measure: "ti" = per-noise-level diffusion timestep '
                        'actually used to perturb the trajectory (recommended); "privacy" = '
                        'global privacy score column of --head_path.')
    p.add_argument('--adapt_norm', type=str, default='snr',
                   choices=['rank', 'relative', 'absolute', 'snr'],
                   help='[adaptive] "relative" = quantile-clipped min-max over the private set '
                        'of the current noise level; "absolute" = s_i = t_i/(n_steps-1), '
                        'comparable across noise levels; "rank" = empirical-CDF rank, immune '
                        'to the heavy right skew of t_i; "snr" uses actual diffusion '
                        'noise-to-signal odds (requires --adapt_key ti).')
    p.add_argument('--adapt_snr_scale', type=float, default=0.1,
                   help='[adaptive, snr] positive k in s=(1-abar)/(1-abar+k*abar). '
                        'Smaller k gives stronger guidance at the SAME t_i. '
                        'This scale is a tuning parameter, not an optimality guarantee.')
    p.add_argument('--head_id_weight', type=float, default=4.0,
                   help='Nonnegative categorical OD-ID mismatch cost in coarse AND fine '
                        'retrieval. Positive values stop treating ID numbers as coordinates. '
                        'Uses existing conditioning IDs and clean-reference IDs only.')
    p.add_argument('--adapt_q', type=float, default=0.05,
                   help='[adaptive, relative] quantile clipped at each end before min-max.')
    p.add_argument('--tau', type=float, default=1.0, help='Softmax temperature (keep >= ~0.5)')
    p.add_argument('--lambda_state', type=float, default=1.0, help='Weight of state distance D_state')
    p.add_argument('--lambda_vel', type=float, default=1.0, help='Weight of direction distance D_vel')
    p.add_argument('--clip_norm', type=float, default=1.0,
                   help='Max L2 norm of eps_retrieval as a multiple of ||eps_theta|| (per sample)')
    # fallback only: privacy_score -> timestep when per-noise training file absent
    p.add_argument('--ti_max', type=int, default=-1,
                   help='Fallback t_i = clamp(privacy_score * ti_max). -1 => num_diffusion_timesteps')
    p.add_argument('--seed', type=int, default=0, help='Seed for the fixed clean noise eps_clean')
    parsed = p.parse_args()
    if not np.isfinite(parsed.adapt_gamma) or parsed.adapt_gamma <= 0:
        p.error('--adapt_gamma must be finite and > 0 for monotone guidance')
    if not 0 <= parsed.lam_ratio_min <= 1:
        p.error('--lam_ratio_min must be in [0, 1]')
    if not np.isfinite(parsed.adapt_snr_scale) or parsed.adapt_snr_scale <= 0:
        p.error('--adapt_snr_scale must be finite and > 0')
    if parsed.adaptive and parsed.adapt_norm == 'snr' and parsed.adapt_key != 'ti':
        p.error('--adapt_norm snr requires --adapt_key ti')
    if not np.isfinite(parsed.head_id_weight) or parsed.head_id_weight < 0:
        p.error('--head_id_weight must be finite and >= 0')
    return parsed


cli_args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.cuda_device

# =========================
# 1. Config
# =========================
temp = {k: SimpleNamespace(**v) for k, v in args.items()}
config = SimpleNamespace(**temp)
if not 1 <= cli_args.timesteps <= config.diffusion.num_diffusion_timesteps:
    raise ValueError('--timesteps must be between 1 and the configured diffusion steps')
if cli_args.batch_size < 1:
    raise ValueError('--batch_size must be positive')

device = "cuda"
L = config.data.traj_length
HEAD_DIM = config.data.head_dim

# =========================
# 2. Diffusion settings
# =========================
n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, n_steps).to(device)

eta = cli_args.eta
timesteps = cli_args.timesteps
skip = n_steps // timesteps
seq = list(range(0, n_steps, skip))
seq_next = [-1] + seq[:-1]

ti_max = n_steps if cli_args.ti_max < 0 else cli_args.ti_max

# precompute alpha_bar for every diffusion step (index by raw timestep i)
_all_t = torch.arange(n_steps, device=device).long()
alpha_bar_all = compute_alpha(beta, _all_t).view(-1)        # [n_steps]


# =========================
# 3. Load trajectories + heads, split clean / private
# =========================
def to_channel_first(traj):
    """Accept [N,Len,2] or [N,2,Len] -> [N,2,Len]."""
    if traj.shape[1] == 2 and traj.shape[2] != 2:
        return traj                              # already [N,2,Len]
    if traj.shape[2] == 2:                        # [N,Len,2]
        return np.transpose(traj, (0, 2, 1))
    raise ValueError(f"Unrecognized traj shape {traj.shape}")


traj = np.load(cli_args.traj_path, allow_pickle=True).astype(np.float32)
traj = to_channel_first(traj)                                # [N,2,Len]
head_np = np.load(cli_args.head_path, allow_pickle=True).astype(np.float32)
assert traj.shape[0] == head_np.shape[0], \
    f"traj/head row mismatch: {traj.shape[0]} vs {head_np.shape[0]}"

priv = head_np[:, -1]
clean_mask = np.abs(priv) <= 1e-8
private_mask = priv > 1e-8
private_idx = np.where(private_mask)[0]                       # row positions in [0,N)

clean_traj = torch.from_numpy(traj[clean_mask]).float().to(device)        # [Nc,2,Len]
clean_feat = torch.from_numpy(head_np[clean_mask, :HEAD_DIM]).float().to(device)   # [Nc,HEAD_DIM]
private_attr = torch.from_numpy(head_np[private_idx, :HEAD_DIM]).float()  # [Np,HEAD_DIM] (cpu)

Nc = clean_traj.shape[0]
Np = private_attr.shape[0]
print(f"clean refs: {Nc}   |   private to generate: {Np}")
assert Nc > 0 and Np > 0, "Need both clean refs and private heads."
assert Nc >= cli_args.top_m >= cli_args.top_k >= 1, \
    f"Require Nc({Nc}) >= top_m({cli_args.top_m}) >= top_k({cli_args.top_k}) >= 1"

# Stage-1 normalization stats from clean head features (z-score on first HEAD_DIM dims)
feat_mean = clean_feat.mean(0, keepdim=True)
feat_std = clean_feat.std(0, keepdim=True).clamp_min(1e-6)
clean_feat_n = (clean_feat - feat_mean) / feat_std                       # [Nc,HEAD_DIM]

# Fixed clean noise eps_clean (sampled ONCE, reused across steps & noise levels).
# Only used to noise clean refs to timestep t for the state/direction DISTANCES.
g = torch.Generator(device='cpu').manual_seed(cli_args.seed)
eps_clean = torch.randn(Nc, 2, L, generator=g).to(device)               # [Nc,2,Len]

# Guidance levels to sweep (each written to its own sub-folder).
if cli_args.guide_lambdas.strip():
    guide_lambdas = [float(s) for s in cli_args.guide_lambdas.split(",") if s.strip()]
else:
    guide_lambdas = [cli_args.guide_lambda]
if not guide_lambdas or any(not np.isfinite(gl) or not 0 <= gl <= 1 for gl in guide_lambdas):
    raise ValueError('Guidance weights/peaks must be finite values in [0,1]')
if cli_args.adaptive:
    print("Guidance levels to generate (ADAPTIVE, value = peak lam_max):", guide_lambdas)
    print(f"  lam_min = {cli_args.lam_ratio_min:g} * lam_max | gamma={cli_args.adapt_gamma:g} "
          f"| key={cli_args.adapt_key} | norm={cli_args.adapt_norm} | q={cli_args.adapt_q:g}")
else:
    print("Guidance levels to generate (FIXED):", guide_lambdas)


# =========================
# 3a. Stage-1 coarse retrieval (level- AND noise-independent -> precompute once)
# =========================
# cand_idx depends only on the private head features vs the clean head features,
# both fixed across noise levels and guidance levels, so we compute it a single
# time here in batches and reuse it for every (noise_level, guide_lambda) pair.
def precompute_cand_idx():
    idx_chunks = []
    for s in range(0, Np, cli_args.batch_size):
        feat = private_attr[s:s + cli_args.batch_size].to(device)
        feat = (feat - feat_mean) / feat_std                              # [b,HEAD_DIM]
        if cli_args.head_id_weight > 0:
            query_ids = private_attr[s:s + cli_args.batch_size, 6:8].to(device)
            mismatch = (query_ids[:, None, :] != clean_feat[None, :, 6:8]).float().sum(-1)
            d_head = torch.cdist(feat[:, :6], clean_feat_n[:, :6]) + cli_args.head_id_weight * mismatch
        else:
            d_head = torch.cdist(feat, clean_feat_n)                      # [b,Nc]
        ci = torch.topk(d_head, cli_args.top_m, largest=False, dim=1).indices
        idx_chunks.append(ci.cpu())
    return torch.cat(idx_chunks, dim=0)                                   # [Np,top_m] (cpu)


cand_idx_all = precompute_cand_idx()
print(f"Precomputed Stage-1 candidates: {tuple(cand_idx_all.shape)}")


# =========================
# 3b. Privacy timestep t_i, consistent with training
# =========================
def col_to_timesteps(col):
    """Map a head last-column to diffusion timesteps, auto-detecting representation.
    - integer-ish / max>1  -> raw diffusion timestep (training's `.long()`)
    - normalized score in [0,1] (non-integer) -> scale by ti_max
    """
    col = np.asarray(col, dtype=np.float64)
    cmax = float(np.nanmax(col))
    integerish = np.allclose(col, np.rint(col), atol=1e-4)
    if cmax <= 1.0 + 1e-4 and not integerish:
        ti = np.rint(np.clip(col, 0.0, 1.0) * ti_max)
        rep = f"normalized privacy score -> * ti_max({ti_max})"
    else:
        ti = np.rint(col)
        rep = "raw diffusion timestep (.long(), matches training)"
    return np.clip(ti, 0, n_steps - 1).astype(np.int64), rep


def load_ti(noise_level):
    """Return (t_i tensor [Np], description) for the given noise level."""
    path = os.path.join(cli_args.train_head_dir, f"noise_{noise_level}", "traj_features.npy")
    if cli_args.train_head_dir and os.path.exists(path):
        h = np.load(path, allow_pickle=True).astype(np.float32)
        assert h.shape[0] == head_np.shape[0], f"train head row mismatch in {path}"
        ti_all, rep = col_to_timesteps(h[:, -1])
        src = f"{path} | {rep}"
    else:
        print(f"[warn] per-noise training head not found ({path}); "
              f"falling back to global privacy score scaling.")
        ti_all, rep = col_to_timesteps(head_np[:, -1])
        src = f"{cli_args.head_path} (fallback) | {rep}"
    return torch.from_numpy(ti_all[private_idx]).long(), src


# =========================
# 3c. Adaptive guidance strength: per-trajectory noisiness -> lambda_i
# =========================
def noisiness_scores(private_ti):
    """Per-private-trajectory noisiness s in [0,1] (larger = noisier).

    Returns (s [Np] float cpu tensor, human-readable description).
    """
    if cli_args.adapt_key == 'ti':
        raw = private_ti.float().numpy()
        raw_desc = 't_i (per-noise diffusion timestep)'
    else:
        raw = head_np[private_idx, -1].astype(np.float64)
        raw_desc = 'privacy score (global head last column)'

    if cli_args.adapt_norm == 'snr':
        # The same t_i gets the same score regardless of the other trajectories.
        # Noise-to-signal odds grow monotonically with t_i for positive beta.
        abar = alpha_bar_all.detach().cpu().numpy()[private_ti.numpy()]
        noise_var = np.maximum(1.0 - abar, 0.0)
        s = noise_var / (noise_var + cli_args.adapt_snr_scale * abar)
        desc = f'diffusion noise odds / (odds + {cli_args.adapt_snr_scale:g}) (snr)'
    elif cli_args.adapt_norm == 'absolute':
        denom = float(n_steps - 1) if cli_args.adapt_key == 'ti' \
            else max(float(np.nanmax(raw)), 1e-8)
        s = np.clip(raw / denom, 0.0, 1.0)
        desc = f'{raw_desc} / {denom:g} (absolute)'
    elif cli_args.adapt_norm == 'rank':
        # Empirical CDF. t_i is heavily right-skewed (median ~ 7% of max), so
        # min-max collapses most trajectories to s~0; ranks spread them evenly
        # and make s depend only on the ORDER of noisiness, not its scale.
        r = rankdata(raw, method='average')
        s = (r - 1.0) / max(len(r) - 1.0, 1.0)
        desc = f'{raw_desc} empirical-CDF rank (rank)'
    else:
        q = float(np.clip(cli_args.adapt_q, 0.0, 0.49))
        lo, hi = np.quantile(raw, q), np.quantile(raw, 1.0 - q)
        if hi - lo < 1e-8:                      # degenerate spread (e.g. noise 0.00)
            s = np.full_like(raw, 0.5, dtype=np.float64)
            desc = f'{raw_desc}: spread ~0 (all == {float(lo):g}) -> s=0.5 for all'
        else:
            s = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
            desc = (f'{raw_desc} min-max on [q{q:g}={float(lo):g}, '
                    f'q{1 - q:g}={float(hi):g}] (relative)')
    return torch.from_numpy(np.asarray(s, dtype=np.float32)), desc


def lambda_vector(s, lam_max):
    """s [.] in [0,1] -> per-sample mixing weight in [lam_min, lam_max]."""
    lam_min = cli_args.lam_ratio_min * lam_max
    lam = lam_min + (lam_max - lam_min) * s.clamp(0.0, 1.0).pow(cli_args.adapt_gamma)
    return lam.clamp(0.0, 1.0)


# =========================
# 4. Retrieval guidance (Stage-2, vectorized over guided samples)
# =========================
def retrieval_noise(x_g, eps_g, cand_idx, i, query_attr=None):
    """
    x_g      : [G,2,Len]  current generated private states (guided subset)
    eps_g    : [G,2,Len]  model predicted noise for those samples
    cand_idx : [G,top_m]  Stage-1 clean candidate indices for those samples
    i        : int        current diffusion timestep
    returns  : [G,2,Len]  eps_retrieval, norm-clipped to clip_norm*||eps_g||
    """
    a = alpha_bar_all[i]
    sa = a.sqrt().clamp_min(1e-6)
    s1 = (1.0 - a).sqrt().clamp_min(1e-6)

    clean_x0 = clean_traj[cand_idx]            # [G,top_m,2,Len]
    eps_cand = eps_clean[cand_idx]             # [G,top_m,2,Len]
    x_t_clean = sa * clean_x0 + s1 * eps_cand  # [G,top_m,2,Len]

    xg = x_g.unsqueeze(1)                       # [G,1,2,Len]
    eg = eps_g.unsqueeze(1)                     # [G,1,2,Len]

    # state distance
    D_state = (xg - x_t_clean).flatten(2).norm(dim=-1)            # [G,top_m]

    # direction distance (1 - cosine of denoising residuals)
    d_clean = clean_x0 - x_t_clean                                # [G,top_m,2,Len]
    x0_hat = (xg - s1 * eg) / sa                                  # [G,1,2,Len]
    d_priv = (x0_hat - xg).expand_as(d_clean)                     # [G,top_m,2,Len]
    cos = F.cosine_similarity(d_priv.flatten(2), d_clean.flatten(2), dim=-1)
    D_vel = 1.0 - cos                                            # [G,top_m]

    D = cli_args.lambda_state * D_state + cli_args.lambda_vel * D_vel   # [G,top_m]
    if cli_args.head_id_weight > 0:
        assert query_attr is not None, 'Categorical retrieval requires the conditioning attributes'
        mismatch = (query_attr[:, None, 6:8] != clean_feat[cand_idx][:, :, 6:8]).float().sum(-1)
        D = D + cli_args.head_id_weight * mismatch

    # top_k smallest distances -> softmax weights
    topv, topi = torch.topk(-D, cli_args.top_k, dim=1)           # [G,top_k]  (topv = -D)
    w = torch.softmax(topv / cli_args.tau, dim=1)               # [G,top_k]

    # --- retrieval noise from the RETRIEVED CLEAN TRAJECTORIES ---
    # x0_retr = weighted clean trajectory; convert to the noise that the current
    # private state x_t would need to denoise toward that clean manifold point.
    clean_x0_topk = torch.gather(
        clean_x0, 1,
        topi.view(*topi.shape, 1, 1).expand(-1, -1, 2, L))      # [G,top_k,2,Len]
    x0_retr = (w.view(*w.shape, 1, 1) * clean_x0_topk).sum(1)    # [G,2,Len]
    eps_retr = (x_g - sa * x0_retr) / s1                         # [G,2,Len]

    # per-sample norm clipping relative to predicted-noise norm
    n_retr = eps_retr.flatten(1).norm(dim=-1).clamp_min(1e-12)   # [G]
    n_pred = eps_g.flatten(1).norm(dim=-1)                       # [G]
    cap = cli_args.clip_norm * n_pred
    scale = torch.minimum(torch.ones_like(n_retr), cap / n_retr)
    eps_retr = eps_retr * scale.view(-1, 1, 1)
    return eps_retr


# =========================
# 5. Loop over noise levels (models) x guidance levels
# =========================
ROOT = cli_args.root


def gl_tag(gl):
    """Compact, filesystem-friendly name for a guidance level (1.0 -> '1')."""
    return "guide_%g" % gl


# Fixed (non-guide) hyper-parameters share one parent folder; each guidance level
# then gets its own sub-folder guide_<level>/ so runs never overwrite each other.
# Override the parent name with --run_tag if desired.
_adapt_tag = (
    f"_adapt-{cli_args.adapt_key}-{cli_args.adapt_norm}"
    f"_rmin{cli_args.lam_ratio_min:g}_g{cli_args.adapt_gamma:g}"
) if cli_args.adaptive else ""
if cli_args.adaptive and cli_args.adapt_norm == 'snr':
    _adapt_tag += f'_snrk{cli_args.adapt_snr_scale:g}'
if cli_args.head_id_weight > 0:
    _adapt_tag += f'_idw{cli_args.head_id_weight:g}'
run_tag = cli_args.run_tag.strip() or (
    f"m{cli_args.top_m}_k{cli_args.top_k}"
    f"_tau{cli_args.tau}_ls{cli_args.lambda_state}_lv{cli_args.lambda_vel}"
    f"_clip{cli_args.clip_norm}{_adapt_tag}_seed{cli_args.seed}"
)
RESULT_ROOT = os.path.join(cli_args.result_root, run_tag)
level_dirs = {gl: os.path.join(RESULT_ROOT, gl_tag(gl)) for gl in guide_lambdas}
for d in level_dirs.values():
    os.makedirs(d, exist_ok=True)
with open(os.path.join(RESULT_ROOT, 'run_config.json'), 'w') as f:
    json.dump(dict(arguments=vars(cli_args), diffusion=vars(config.diffusion),
                   private_count=Np, clean_count=Nc, uniform_per_trajectory=True), f, indent=2)
np.save(os.path.join(RESULT_ROOT, 'private_indices.npy'), private_idx)
print(f"Output parent: {RESULT_ROOT}")
for gl, d in level_dirs.items():
    label = "lam_max" if cli_args.adaptive else "guide_lambda"
    print(f"  {label}={gl:g} -> {d}")

noise_dirs = sorted(d for d in os.listdir(ROOT) if d.startswith(cli_args.noise_prefix))
if cli_args.noise_levels.strip():
    wanted = {s.strip() for s in cli_args.noise_levels.split(",") if s.strip()}
    noise_dirs = [d for d in noise_dirs if d.split("noise_")[1].split("_")[0] in wanted]
    missing = wanted - {d.split('noise_')[1].split('_')[0] for d in noise_dirs}
    if missing:
        raise FileNotFoundError(f'Missing requested model noise levels under {ROOT}: {sorted(missing)}')
if not noise_dirs:
    raise FileNotFoundError(f'No model directories matching {cli_args.noise_prefix} under {ROOT}')
print("Processing noise levels:", noise_dirs)

idx_all = torch.arange(Np)

for noise_dir in noise_dirs:
    noise_level = noise_dir.split("noise_")[1].split("_")[0]
    print(f"\n===== {noise_dir} =====")

    model_root = os.path.join(ROOT, noise_dir, "models")
    time_dirs = sorted(os.listdir(model_root))
    assert time_dirs, f"No model dirs in {model_root}"
    model_path = os.path.join(model_root, time_dirs[-1], cli_args.ckpt_name)
    assert os.path.exists(model_path), f"Model not found: {model_path}"
    print("Loading model:", model_path)

    unet = Guide_UNet(config).to(device)
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.eval()

    # per-noise-level t_i (training-consistent timestep representation)
    private_ti, ti_src = load_ti(noise_level)
    print(f"t_i source: {ti_src}  | t_i[min,max]=({int(private_ti.min())},{int(private_ti.max())})")

    # per-trajectory noisiness -> adaptive strength (constant within a noise level)
    if cli_args.adaptive:
        private_s, s_src = noisiness_scores(private_ti)
        print(f"adaptive noisiness: {s_src}")
        for gl in guide_lambdas:
            lv = lambda_vector(private_s, gl)
            print(f"  lam_max={gl:g} -> lambda_i in [{float(lv.min()):.3f}, "
                  f"{float(lv.max()):.3f}], mean {float(lv.mean()):.3f}")
    else:
        private_s = torch.zeros(Np, dtype=torch.float32)

    loader = DataLoader(TensorDataset(private_attr, private_ti, private_s, idx_all),
                        batch_size=cli_args.batch_size, shuffle=False)

    # one output list per guidance level for this noise level
    gen_by_level = {gl: [] for gl in guide_lambdas}

    for b_i, (attr_b, ti_b, s_b, idx_b) in enumerate(
            tqdm(loader, desc=f"Generating ({noise_level})")):
        attr_b = attr_b.to(device)                    # [B,HEAD_DIM] conditioning (matches training)
        ti_b = ti_b.to(device)                        # [B]
        s_b = s_b.to(device)                          # [B] noisiness in [0,1]
        cand_idx = cand_idx_all[idx_b].to(device)     # [B,top_m] precomputed Stage-1
        B = attr_b.shape[0]

        # Shared initial noise across guidance levels, so differences between
        # levels are attributable to the guidance strength alone (and runs are
        # reproducible via --seed).
        g_init = torch.Generator(device='cpu').manual_seed(cli_args.seed + 100000 + b_i)
        x0_init = torch.randn(B, 2, L, generator=g_init).to(device)

        for gl in guide_lambdas:
            # fixed mode: scalar weight; adaptive mode: one weight per trajectory
            lam_b = lambda_vector(s_b, gl) if cli_args.adaptive else None
            x = x0_init.clone()
            ims = []
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full((B,), i, device=device)
                next_t = torch.full((B,), j, device=device)

                with torch.no_grad():
                    pred_noise = unet(x, t, attr_b)

                    guide_mask = (i <= ti_b)                          # [B] bool
                    if guide_mask.any():
                        gi = guide_mask.nonzero(as_tuple=True)[0]     # [G]
                        eps_retr = retrieval_noise(x[gi], pred_noise[gi], cand_idx[gi], i, attr_b[gi])
                        lam_g = gl if lam_b is None else lam_b[gi].view(-1, 1, 1)
                        pred_noise[gi] = (1.0 - lam_g) * pred_noise[gi] + lam_g * eps_retr

                    x = p_xt(x, pred_noise, t, next_t, beta, eta)
                    if i % 10 == 0:
                        ims.append(x.cpu())

            trajs = ims[-1].numpy()[:, :2, :]
            for b in range(B):
                gen_by_level[gl].append(trajs[b].T.astype(float))

    # save one pkl per guidance level for this noise level
    for gl in guide_lambdas:
        save_path = os.path.join(level_dirs[gl], f"Gen_traj_noise_{noise_level}_guided.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(gen_by_level[gl], f)
        if cli_args.adaptive:
            lam_path = os.path.join(level_dirs[gl], f"guide_lambda_noise_{noise_level}.npy")
            np.save(lam_path, lambda_vector(private_s, gl).numpy())
            print(f"Saved -> {save_path}  ({len(gen_by_level[gl])} trajs, "
                  f"adaptive lam_max={gl:g}; per-traj lambdas -> {lam_path})")
        else:
            print(f"Saved -> {save_path}  ({len(gen_by_level[gl])} trajs, guide_lambda={gl:g})")

print("\n🎉 All noise levels x guidance levels finished (retrieval-guided).")
