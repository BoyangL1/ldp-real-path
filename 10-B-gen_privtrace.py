"""
Baseline 10-B: generate trajectories from fitted PrivTrace models (9-B),
one result per noise level - mirroring 5-A / 5-B.

Loads Baselines_nagoya/privtrace/noise_{nl}/markov.pkl (fitted two-level
grid + Markov statistics saved by 9-B) and samples synthetic trajectories.
No training data is touched at generation time. Outputs are NOT row-aligned
with the real private set - distribution-level metrics only.

Usage
-----
  python 10-B-gen_privtrace.py                            # all fitted levels
  python 10-B-gen_privtrace.py --noise_levels 0.50 --n_samples 10000 --seed 1
"""
import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate from fitted PrivTrace models")
    p.add_argument('--head_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_features.npy')
    p.add_argument('--stats_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_tensor_stats.npz')
    p.add_argument('--model_root', type=str, default='./Baselines_nagoya/privtrace')
    p.add_argument('--result_root', type=str, default='./Baseline_result_nagoya/privtrace')
    p.add_argument('--noise_levels', type=str, default='',
                   help='Comma-separated subset, e.g. "0.00,0.50". Empty = all fitted.')
    p.add_argument('--n_samples', type=int, default=0,
                   help='Synthetic trajectories per level (0 = #private rows)')
    p.add_argument('--max_walk', type=int, default=40)
    p.add_argument('--target_len', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


cli = parse_args()
rng = np.random.default_rng(cli.seed)


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


def seq_to_traj(seq, centers, sizes, target_len):
    pts = centers[seq] + (rng.random((len(seq), 2)) - .5) * sizes[seq]
    if len(pts) == 1:
        return np.repeat(pts, target_len, axis=0)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    if cum[-1] < 1e-9:
        return np.repeat(pts[:1], target_len, axis=0)
    u = np.linspace(0, cum[-1], target_len)
    return np.stack([np.interp(u, cum, pts[:, 0]),
                     np.interp(u, cum, pts[:, 1])], axis=1)


stats = np.load(cli.stats_path)
MEAN, STD = stats['mean'].reshape(1, 2), stats['std'].reshape(1, 2)

head_global = np.load(cli.head_path, allow_pickle=True).astype(np.float32)
n_private = int((head_global[:, -1] > 1e-8).sum())
n_samples = cli.n_samples if cli.n_samples > 0 else n_private
print(f"Synthesizing {n_samples} trajectories per level")

noise_dirs = sorted(d for d in os.listdir(cli.model_root) if d.startswith('noise_'))
if cli.noise_levels.strip():
    wanted = {s.strip() for s in cli.noise_levels.split(',') if s.strip()}
    noise_dirs = [d for d in noise_dirs if d.split('noise_')[1] in wanted]
assert noise_dirs, f"No fitted noise_* dirs found under {cli.model_root}"
print("Found fitted levels:", noise_dirs)

os.makedirs(cli.result_root, exist_ok=True)

for noise_dir in noise_dirs:
    nl = noise_dir.split('noise_')[1]
    model_path = os.path.join(cli.model_root, noise_dir, 'markov.pkl')
    assert os.path.exists(model_path), f"Missing model: {model_path}"
    print(f"\n===== PrivTrace gen | noise {nl} =====")

    with open(model_path, 'rb') as f:
        m = pickle.load(f)

    Gen_traj = []
    for _ in tqdm(range(n_samples), desc=f"synth ({nl})"):
        seq = sample_seq(m['start'], m['t1'], m['t2'], m['END'], cli.max_walk)
        pts = seq_to_traj(seq, m['centers'], m['sizes'], cli.target_len)
        Gen_traj.append(((pts - MEAN) / STD).astype(float))

    save_path = os.path.join(cli.result_root, f'Gen_traj_noise_{nl}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(Gen_traj, f)
    print(f"Saved {len(Gen_traj)} trajectories to {save_path}")

print("\n🎉 All levels generated (PrivTrace). NOTE: not row-aligned; "
      "distribution-level metrics only.")
