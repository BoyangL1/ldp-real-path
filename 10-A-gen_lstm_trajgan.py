"""
Baseline 10-A: generate trajectories from trained LSTM-TrajGAN checkpoints
(9-A), one result per noise level - mirroring 5-A / 5-B.

For every Baselines_nagoya/lstm_trajgan/noise_{nl}/G.pt this loads the
generator, feeds it the LDP-noised private trajectories of the SAME noise
level (protocol input) plus fresh latent noise, and saves the row-aligned
synthetic private set to <result_root>/Gen_traj_noise_{nl}.pkl.

Usage
-----
  python 10-A-gen_lstm_trajgan.py                         # all trained levels
  python 10-A-gen_lstm_trajgan.py --noise_levels 0.50 --seed 1
"""
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Generate from trained LSTM-TrajGAN checkpoints")
    p.add_argument('--sweep_root', type=str,
                   default='data/traj_privacy/nagoya/noise_sweep')
    p.add_argument('--head_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_features.npy')
    p.add_argument('--model_root', type=str, default='./Baselines_nagoya/lstm_trajgan')
    p.add_argument('--result_root', type=str, default='./Baseline_result_nagoya/lstm_trajgan')
    p.add_argument('--noise_levels', type=str, default='',
                   help='Comma-separated subset, e.g. "0.00,0.50". Empty = all trained.')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--latent_dim', type=int, default=100,
                   help='Must match 9-A training')
    p.add_argument('--hidden_dim', type=int, default=100,
                   help='Must match 9-A training')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda_device', type=str, default='0')
    return p.parse_args()


cli = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = cli.cuda_device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cli.seed)
np.random.seed(cli.seed)

SLOTS_PER_DAY = 288


# ---- model definitions: MUST match 9-A exactly (state-dict compatible) ----
class PointFusion(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.sp = nn.Linear(2, out_dim)
        self.hr = nn.Linear(24, out_dim)
        self.fuse = nn.Linear(2 * out_dim, out_dim)

    def forward(self, dev, hour_oh):
        h = self.hr(hour_oh).unsqueeze(1).expand(-1, dev.size(1), -1)
        s = self.sp(dev)
        return torch.relu(self.fuse(torch.cat([s, h], dim=-1)))


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super().__init__()
        self.fusion = PointFusion(64)
        self.inp = nn.Linear(64 + latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 2)

    def forward(self, dev, hour_oh, z):
        f = self.fusion(dev, hour_oh)
        zz = z.unsqueeze(1).expand(-1, f.size(1), -1)
        h = torch.relu(self.inp(torch.cat([f, zz], dim=-1)))
        h, _ = self.lstm(h)
        return torch.tanh(self.out(h)) * 3.0


def hour_onehot(departure_slot):
    hour = (departure_slot.astype(int) * 24 // SLOTS_PER_DAY) % 24
    oh = np.zeros((len(hour), 24), dtype=np.float32)
    oh[np.arange(len(hour)), hour] = 1.0
    return oh


# ---- main loop over trained noise levels (5-A style) ----
head_global = np.load(cli.head_path, allow_pickle=True).astype(np.float32)
priv_idx = np.where(head_global[:, -1] > 1e-8)[0]
print(f"Private rows to generate: {len(priv_idx)}")

noise_dirs = sorted(d for d in os.listdir(cli.model_root) if d.startswith('noise_'))
if cli.noise_levels.strip():
    wanted = {s.strip() for s in cli.noise_levels.split(',') if s.strip()}
    noise_dirs = [d for d in noise_dirs if d.split('noise_')[1] in wanted]
assert noise_dirs, f"No trained noise_* dirs found under {cli.model_root}"
print("Found trained levels:", noise_dirs)

os.makedirs(cli.result_root, exist_ok=True)

for noise_dir in noise_dirs:
    nl = noise_dir.split('noise_')[1]
    ckpt = os.path.join(cli.model_root, noise_dir, 'G.pt')
    assert os.path.exists(ckpt), f"Missing checkpoint: {ckpt}"
    print(f"\n===== LSTM-TrajGAN gen | noise {nl} =====")

    traj = np.load(os.path.join(cli.sweep_root, f'noise_{nl}', 'traj.npy')).astype(np.float32)
    feats = np.load(os.path.join(cli.sweep_root, f'noise_{nl}', 'traj_features.npy')).astype(np.float32)
    centroid = traj.mean(axis=1, keepdims=True)
    dev = traj - centroid
    hour_oh = hour_onehot(feats[:, 0])

    G = Generator(cli.latent_dim, cli.hidden_dim).to(device)
    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()

    Gen_traj = []
    with torch.no_grad():
        for start in tqdm(range(0, len(priv_idx), cli.batch_size), desc=f"gen ({nl})"):
            ids = priv_idx[start:start + cli.batch_size]
            dev_b = torch.from_numpy(dev[ids]).to(device)
            hr_b = torch.from_numpy(hour_oh[ids]).to(device)
            z = torch.randn(len(ids), cli.latent_dim, device=device)
            fake = G(dev_b, hr_b, z).cpu().numpy() + centroid[ids]
            for b in range(len(ids)):
                Gen_traj.append(fake[b].astype(float))

    save_path = os.path.join(cli.result_root, f'Gen_traj_noise_{nl}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(Gen_traj, f)
    print(f"Saved {len(Gen_traj)} trajectories to {save_path}")

print("\n🎉 All levels generated (LSTM-TrajGAN).")
