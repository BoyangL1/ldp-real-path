"""
Baseline 9-A: LSTM-TrajGAN (PyTorch adaptation) on the Nagoya LDP noise sweep.

Reference
---------
Rao, Gao, Kang, Liang: "LSTM-TrajGAN: A Deep Learning Approach to
Trajectory Privacy Protection" (GIScience 2020). Original code is
TensorFlow/Keras (https://github.com/GeoDS/LSTM-TrajGAN); this is a faithful
PyTorch adaptation for our fixed-length (lat,lon) grid trajectories.

Protocol (identical input to LDP-DiffTraj training)
---------------------------------------------------
Per noise level nl the model sees ONLY noise_sweep/noise_{nl}/traj.npy
(clean privacy_score==0 subset + LDP-noised private subset, z-scored grid
coords) plus the same conditioning head (departure slot). No clean private
trajectory is ever read.

Adaptations vs the original (documented for the paper)
------------------------------------------------------
  * Points carry no per-point timestamp / POI category in our data, so the
    temporal channel is the trajectory-level departure hour (one-hot 24,
    broadcast to every point); the category channel is dropped.
  * Spatial input is the per-point deviation from the trajectory centroid,
    exactly as in the original (they center on the dataset centroid; we use
    the per-trajectory centroid which is available at generation time from
    the noised trajectory itself).
  * TrajLoss = BCE(adversarial) + w_sp * MSE(spatial deviations), i.e. the
    original's alpha=1, beta=10 spatial term; categorical terms dropped.

Generation: one synthetic trajectory per private (privacy_score>0) row,
row-aligned with the 5-A outputs, saved as
    <result_root>/Gen_traj_noise_{nl}.pkl   (list of [Len,2] float arrays,
                                             z-scored grid space)

Usage
-----
  python 9-A-baseline_lstm_trajgan.py                     # all noise levels
  python 9-A-baseline_lstm_trajgan.py --noise_levels 0.50 --epochs 2 \
      --max_train 512                                     # smoke test
"""
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =========================
# 0. CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="LSTM-TrajGAN baseline on the LDP noise sweep")
    p.add_argument('--sweep_root', type=str,
                   default='data/traj_privacy/nagoya/noise_sweep',
                   help='Dir with noise_{nl}/traj.npy + traj_features.npy')
    p.add_argument('--head_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_features.npy',
                   help='Global head: last col = privacy score (clean/private split)')
    p.add_argument('--model_root', type=str, default='./Baselines_nagoya/lstm_trajgan')
    p.add_argument('--result_root', type=str, default='./Baseline_result_nagoya/lstm_trajgan')
    p.add_argument('--noise_levels', type=str, default='',
                   help='Comma-separated subset, e.g. "0.00,0.50". Empty = all.')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--latent_dim', type=int, default=100)
    p.add_argument('--hidden_dim', type=int, default=100)
    p.add_argument('--w_sp', type=float, default=10.0,
                   help='Weight of the spatial MSE term in TrajLoss (paper: 10)')
    p.add_argument('--max_train', type=int, default=0,
                   help='Subsample training rows for smoke tests (0 = all)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda_device', type=str, default='0')
    return p.parse_args()


cli = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = cli.cuda_device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cli.seed)
np.random.seed(cli.seed)

SLOTS_PER_DAY = 288  # departure slot resolution (5 min)


# =========================
# 1. Model
# =========================
class PointFusion(nn.Module):
    """Per-point feature fusion (spatial deviation + broadcast hour one-hot)."""

    def __init__(self, out_dim=64):
        super().__init__()
        self.sp = nn.Linear(2, out_dim)
        self.hr = nn.Linear(24, out_dim)
        self.fuse = nn.Linear(2 * out_dim, out_dim)

    def forward(self, dev, hour_oh):
        # dev [B,L,2], hour_oh [B,24]
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
        # z [B,latent] broadcast to every point, as in the original
        f = self.fusion(dev, hour_oh)                                # [B,L,64]
        zz = z.unsqueeze(1).expand(-1, f.size(1), -1)
        h = torch.relu(self.inp(torch.cat([f, zz], dim=-1)))
        h, _ = self.lstm(h)
        return torch.tanh(self.out(h)) * 3.0                         # deviations, z-scored units


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fusion = PointFusion(64)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, dev, hour_oh):
        h, _ = self.lstm(self.fusion(dev, hour_oh))
        return self.out(h[:, -1])                                    # logits


# =========================
# 2. Data helpers
# =========================
def load_level(sweep_root, nl_tag):
    traj = np.load(os.path.join(sweep_root, f'noise_{nl_tag}', 'traj.npy')).astype(np.float32)
    feats = np.load(os.path.join(sweep_root, f'noise_{nl_tag}', 'traj_features.npy')).astype(np.float32)
    return traj, feats


def hour_onehot(departure_slot):
    hour = (departure_slot.astype(int) * 24 // SLOTS_PER_DAY) % 24
    oh = np.zeros((len(hour), 24), dtype=np.float32)
    oh[np.arange(len(hour)), hour] = 1.0
    return oh


# =========================
# 3. Main loop over noise levels
# =========================
head_global = np.load(cli.head_path, allow_pickle=True).astype(np.float32)
priv_idx = np.where(head_global[:, -1] > 1e-8)[0]
print(f"Total rows: {len(head_global)}  |  private rows to generate: {len(priv_idx)}")

level_dirs = sorted(d for d in os.listdir(cli.sweep_root) if d.startswith('noise_'))
levels = [d.split('noise_')[1] for d in level_dirs]
if cli.noise_levels.strip():
    wanted = {s.strip() for s in cli.noise_levels.split(',') if s.strip()}
    levels = [nl for nl in levels if nl in wanted]
    assert levels, f"None of the requested noise levels {wanted} found under {cli.sweep_root}"
print("Processing noise levels:", levels)

os.makedirs(cli.result_root, exist_ok=True)

for nl in levels:
    print(f"\n===== LSTM-TrajGAN | noise {nl} =====")
    traj, feats = load_level(cli.sweep_root, nl)
    assert len(traj) == len(head_global), "sweep file rows must align with the global head"

    centroid = traj.mean(axis=1, keepdims=True)                      # [N,1,2]
    dev = traj - centroid                                            # deviations
    hour_oh = hour_onehot(feats[:, 0])

    train_ids = np.arange(len(traj))
    if cli.max_train > 0 and cli.max_train < len(train_ids):
        train_ids = np.random.choice(train_ids, cli.max_train, replace=False)

    ds = TensorDataset(torch.from_numpy(dev[train_ids]),
                       torch.from_numpy(hour_oh[train_ids]))
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=True, drop_last=True)

    G = Generator(cli.latent_dim, cli.hidden_dim).to(device)
    D = Discriminator(cli.hidden_dim).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=cli.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=cli.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, cli.epochs + 1):
        g_losses, d_losses = [], []
        for dev_b, hr_b in dl:
            dev_b, hr_b = dev_b.to(device), hr_b.to(device)
            B = dev_b.size(0)
            z = torch.randn(B, cli.latent_dim, device=device)
            fake = G(dev_b, hr_b, z)

            # --- D step
            optD.zero_grad()
            d_real = D(dev_b, hr_b)
            d_fake = D(fake.detach(), hr_b)
            d_loss = bce(d_real, torch.ones_like(d_real)) + \
                     bce(d_fake, torch.zeros_like(d_fake))
            d_loss.backward()
            optD.step()

            # --- G step (TrajLoss: adversarial + spatial similarity)
            optG.zero_grad()
            d_fake = D(fake, hr_b)
            g_loss = bce(d_fake, torch.ones_like(d_fake)) + \
                     cli.w_sp * F.mse_loss(fake, dev_b)
            g_loss.backward()
            optG.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        if epoch % 10 == 0 or epoch == cli.epochs or epoch == 1:
            print(f"  epoch {epoch:4d} | G {np.mean(g_losses):.4f} | D {np.mean(d_losses):.4f}")

    model_dir = os.path.join(cli.model_root, f'noise_{nl}')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(model_dir, 'G.pt'))
    torch.save(D.state_dict(), os.path.join(model_dir, 'D.pt'))

    # ---- generation: one synthetic trajectory per private row (row-aligned)
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

print("\n🎉 LSTM-TrajGAN baseline finished.")
