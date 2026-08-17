"""
Baseline 9-B: PrivTrace (adapted) on the Nagoya LDP noise sweep.

Reference
---------
Wang et al.: "PrivTrace: Differentially Private Trajectory Synthesis by
Adaptive Markov Models" (USENIX Security 2023). Self-contained adaptation:
two-level adaptive grid discretization + first/second-order Markov model +
synthesis, following the paper's pipeline.

Protocol (identical input to LDP-DiffTraj training)
---------------------------------------------------
Per noise level nl the model sees ONLY noise_sweep/noise_{nl}/traj.npy.
Privacy is already provided at the input by the LDP perturbation, so the
internal DP noise of PrivTrace is OFF by default (--eps 0 = disabled); pass
--eps to additionally noise the Markov statistics as in the original paper.

Adaptations vs the original (documented for the paper)
------------------------------------------------------
  * Grid sizing: the paper picks level-1/level-2 resolutions from the DP
    budget; with DP off we expose them as CLI parameters (defaults chosen to
    give a comparable number of states on the 100x100 Nagoya grid).
  * PrivTrace is UNCONDITIONAL: it cannot use the head features and its
    output has NO row alignment with the real private set. Use
    distribution-level metrics only (density SSIM / JSD / distance dists).

Output: <result_root>/Gen_traj_noise_{nl}.pkl — list of [Len,2] float arrays
in the SAME z-scored grid space as all other methods (count = #private rows).

Usage
-----
  python 9-B-baseline_privtrace.py                        # all noise levels
  python 9-B-baseline_privtrace.py --noise_levels 0.50 --n_samples 500  # smoke
"""
import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


# =========================
# 0. CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="PrivTrace (adapted) baseline on the LDP noise sweep")
    p.add_argument('--sweep_root', type=str,
                   default='data/traj_privacy/nagoya/noise_sweep')
    p.add_argument('--head_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_features.npy')
    p.add_argument('--stats_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_tensor_stats.npz',
                   help='z-score mean/std used to map to/from grid coordinates')
    p.add_argument('--model_root', type=str, default='./Baselines_nagoya/privtrace',
                   help='Where to save the fitted grid+Markov model per noise level '
                        '(for later generation via 10-B)')
    p.add_argument('--result_root', type=str, default='./Baseline_result_nagoya/privtrace')
    p.add_argument('--noise_levels', type=str, default='',
                   help='Comma-separated subset, e.g. "0.00,0.50". Empty = all.')
    p.add_argument('--g1', type=int, default=16,
                   help='Level-1 grid resolution (G1 x G1 over the bbox)')
    p.add_argument('--g2', type=int, default=4,
                   help='Level-2 subdivision of dense level-1 cells (G2 x G2)')
    p.add_argument('--divide_factor', type=float, default=4.0,
                   help='Subdivide a level-1 cell if its point count exceeds '
                        'divide_factor * (total_points / G1^2)')
    p.add_argument('--eps', type=float, default=0.0,
                   help='Optional Laplace budget on Markov statistics (0 = off; '
                        'privacy is already provided by the LDP input)')
    p.add_argument('--n_samples', type=int, default=0,
                   help='Synthetic trajectories per level (0 = #private rows)')
    p.add_argument('--max_walk', type=int, default=40,
                   help='Max Markov steps per synthetic trajectory')
    p.add_argument('--target_len', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


cli = parse_args()
rng = np.random.default_rng(cli.seed)


# =========================
# 1. Two-level adaptive grid
# =========================
class AdaptiveGrid:
    """Level-1 uniform grid; dense cells subdivided into G2 x G2 (paper Sec 4.1)."""

    def __init__(self, pts, g1, g2, divide_factor):
        self.g1, self.g2 = g1, g2
        self.x_min, self.y_min = pts.min(axis=0) - 1e-6
        self.x_max, self.y_max = pts.max(axis=0) + 1e-6
        self.dx = (self.x_max - self.x_min) / g1
        self.dy = (self.y_max - self.y_min) / g1

        c1 = self._level1(pts)
        counts = np.bincount(c1, minlength=g1 * g1)
        thresh = divide_factor * len(pts) / (g1 * g1)
        self.divided = counts > thresh                               # [g1*g1] bool

        # state ids: undivided level-1 cell -> 1 state; divided -> g2*g2 states
        self.state_of_cell = np.full(g1 * g1, -1, dtype=np.int64)
        self.sub_base = np.full(g1 * g1, -1, dtype=np.int64)
        s = 0
        for c in range(g1 * g1):
            if self.divided[c]:
                self.sub_base[c] = s
                s += g2 * g2
            else:
                self.state_of_cell[c] = s
                s += 1
        self.n_states = s

        # state -> center coordinates (and cell size for jitter)
        self.centers = np.zeros((s, 2), dtype=np.float64)
        self.sizes = np.zeros((s, 2), dtype=np.float64)
        for c in range(g1 * g1):
            cx, cy = c % g1, c // g1
            x0, y0 = self.x_min + cx * self.dx, self.y_min + cy * self.dy
            if self.divided[c]:
                sdx, sdy = self.dx / g2, self.dy / g2
                for sub in range(g2 * g2):
                    sx, sy = sub % g2, sub // g2
                    sid = self.sub_base[c] + sub
                    self.centers[sid] = (x0 + (sx + .5) * sdx, y0 + (sy + .5) * sdy)
                    self.sizes[sid] = (sdx, sdy)
            else:
                sid = self.state_of_cell[c]
                self.centers[sid] = (x0 + .5 * self.dx, y0 + .5 * self.dy)
                self.sizes[sid] = (self.dx, self.dy)

    def _level1(self, pts):
        ix = np.clip(((pts[:, 0] - self.x_min) / self.dx).astype(int), 0, self.g1 - 1)
        iy = np.clip(((pts[:, 1] - self.y_min) / self.dy).astype(int), 0, self.g1 - 1)
        return iy * self.g1 + ix

    def states(self, pts):
        c1 = self._level1(pts)
        out = np.empty(len(pts), dtype=np.int64)
        undiv = ~self.divided[c1]
        out[undiv] = self.state_of_cell[c1[undiv]]
        div = ~undiv
        if div.any():
            c = c1[div]
            cx, cy = c % self.g1, c // self.g1
            x0 = self.x_min + cx * self.dx
            y0 = self.y_min + cy * self.dy
            sx = np.clip(((pts[div, 0] - x0) / (self.dx / self.g2)).astype(int), 0, self.g2 - 1)
            sy = np.clip(((pts[div, 1] - y0) / (self.dy / self.g2)).astype(int), 0, self.g2 - 1)
            out[div] = self.sub_base[c] + sy * self.g2 + sx
        return out


# =========================
# 2. Markov model
# =========================
def lap(shape, scale):
    return rng.laplace(0.0, scale, size=shape) if scale > 0 else 0.0


def build_markov(seqs, n_states, eps):
    """Start dist, 1st-order transitions, 2nd-order transitions, END state."""
    END = n_states
    scale = (3.0 / eps) if eps > 0 else 0.0   # naive 3-way budget split as in the paper

    start = np.zeros(n_states)
    t1 = np.zeros((n_states, n_states + 1))
    t2 = {}
    for s in seqs:
        start[s[0]] += 1
        path = list(s) + [END]
        for a, b in zip(path[:-1], path[1:]):
            t1[a, b] += 1
        for a, b, c in zip(path[:-2], path[1:-1], path[2:]):
            t2.setdefault((a, b), np.zeros(n_states + 1))[c] += 1

    start = np.maximum(start + lap(start.shape, scale), 0)
    start = start / max(start.sum(), 1e-12)
    t1 = np.maximum(t1 + lap(t1.shape, scale), 0)
    t1_sum = t1.sum(axis=1, keepdims=True)
    t1 = np.divide(t1, np.maximum(t1_sum, 1e-12))
    for k in t2:
        v = np.maximum(t2[k] + lap(t2[k].shape, scale), 0)
        t2[k] = v / max(v.sum(), 1e-12)
    return start, t1, t2, END


def sample_seq(start, t1, t2, END, max_walk):
    s0 = rng.choice(len(start), p=start)
    seq = [s0]
    while len(seq) < max_walk:
        key = (seq[-2], seq[-1]) if len(seq) >= 2 else None
        p = t2.get(key) if key is not None else None
        if p is None or p.sum() < 1e-9:
            p = t1[seq[-1]]
        if p.sum() < 1e-9:
            break
        nxt = rng.choice(len(p), p=p / p.sum())
        if nxt == END:
            break
        seq.append(nxt)
    return seq


# =========================
# 3. Sequence -> fixed-length trajectory
# =========================
def seq_to_traj(seq, grid, target_len):
    pts = grid.centers[seq] + (rng.random((len(seq), 2)) - .5) * grid.sizes[seq]
    if len(pts) == 1:
        return np.repeat(pts, target_len, axis=0)
    # resample the polyline at target_len evenly spaced arc-length positions
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    if cum[-1] < 1e-9:
        return np.repeat(pts[:1], target_len, axis=0)
    u = np.linspace(0, cum[-1], target_len)
    x = np.interp(u, cum, pts[:, 0])
    y = np.interp(u, cum, pts[:, 1])
    return np.stack([x, y], axis=1)


# =========================
# 4. Main loop over noise levels
# =========================
stats = np.load(cli.stats_path)
MEAN, STD = stats['mean'].reshape(1, 1, 2), stats['std'].reshape(1, 1, 2)

head_global = np.load(cli.head_path, allow_pickle=True).astype(np.float32)
n_private = int((head_global[:, -1] > 1e-8).sum())
n_samples = cli.n_samples if cli.n_samples > 0 else n_private
print(f"Private rows: {n_private}  |  synthesizing {n_samples} per level")

level_dirs = sorted(d for d in os.listdir(cli.sweep_root) if d.startswith('noise_'))
levels = [d.split('noise_')[1] for d in level_dirs]
if cli.noise_levels.strip():
    wanted = {s.strip() for s in cli.noise_levels.split(',') if s.strip()}
    levels = [nl for nl in levels if nl in wanted]
    assert levels, f"None of the requested noise levels {wanted} found under {cli.sweep_root}"
print("Processing noise levels:", levels)

os.makedirs(cli.result_root, exist_ok=True)

for nl in levels:
    print(f"\n===== PrivTrace | noise {nl} =====")
    traj_z = np.load(os.path.join(cli.sweep_root, f'noise_{nl}', 'traj.npy')).astype(np.float64)
    traj = traj_z * STD + MEAN                                       # back to grid coords

    pts = traj.reshape(-1, 2)
    grid = AdaptiveGrid(pts, cli.g1, cli.g2, cli.divide_factor)
    print(f"  states: {grid.n_states} "
          f"({int(grid.divided.sum())}/{cli.g1 * cli.g1} level-1 cells divided)")

    # trajectories -> state sequences (collapse consecutive duplicates)
    st = grid.states(pts).reshape(len(traj), -1)
    seqs = []
    for row in st:
        s = [row[0]]
        for v in row[1:]:
            if v != s[-1]:
                s.append(int(v))
        seqs.append(s)

    start, t1, t2, END = build_markov(seqs, grid.n_states, cli.eps)

    # persist the fitted model as plain arrays (class-free, loadable by 10-B)
    model_dir = os.path.join(cli.model_root, f'noise_{nl}')
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'markov.pkl'), 'wb') as f:
        pickle.dump({'centers': grid.centers, 'sizes': grid.sizes,
                     'start': start, 't1': t1, 't2': t2, 'END': END,
                     'n_states': grid.n_states}, f)

    Gen_traj = []
    for _ in tqdm(range(n_samples), desc=f"synth ({nl})"):
        seq = sample_seq(start, t1, t2, END, cli.max_walk)
        pts_out = seq_to_traj(seq, grid, cli.target_len)
        Gen_traj.append(((pts_out - MEAN[0]) / STD[0]).astype(float))  # back to z-scored space

    save_path = os.path.join(cli.result_root, f'Gen_traj_noise_{nl}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(Gen_traj, f)
    print(f"Saved {len(Gen_traj)} trajectories to {save_path}")

print("\n🎉 PrivTrace baseline finished. NOTE: outputs are NOT row-aligned; "
      "use distribution-level metrics only.")
