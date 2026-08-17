"""
Baseline 10-C: generate trajectories from trained ControlTraj-style
checkpoints (9-C), one result per noise level - mirroring 5-A / 5-B.

For every Baselines_nagoya/controltraj/noise_{nl}/model_*.pt this loads the
newest (or --ckpt_name) checkpoint, conditions on the private heads and the
cached OSM routes of the SAME noise level (built by 9-C, protocol input),
runs DDIM and saves the row-aligned synthetic private set to
<result_root>/Gen_traj_noise_{nl}.pkl.

Requires the route cache from the 9-C training run (routes_noise_{nl}.npy +
routes_meta.json under --osm_dir); no network access is needed here.

Usage
-----
  python 10-C-gen_controltraj.py                          # all trained levels
  python 10-C-gen_controltraj.py --noise_levels 0.50 --timesteps 100
"""
import argparse
import json
import os
import pickle
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.config import args
from utils.Traj_UNet import Model, WideAndDeep
from utils.utils import p_xt


def parse_args():
    p = argparse.ArgumentParser(description="Generate from trained ControlTraj checkpoints")
    p.add_argument('--head_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_features.npy')
    p.add_argument('--osm_dir', type=str, default='data/osm_routes/nagoya',
                   help='Route cache dir written by 9-C (routes_noise_{nl}.npy + meta)')
    p.add_argument('--model_root', type=str, default='./Baselines_nagoya/controltraj')
    p.add_argument('--result_root', type=str, default='./Baseline_result_nagoya/controltraj')
    p.add_argument('--noise_levels', type=str, default='',
                   help='Comma-separated subset, e.g. "0.00,0.50". Empty = all trained.')
    p.add_argument('--ckpt_name', type=str, default='',
                   help='Checkpoint filename inside noise_{nl}/ (empty = newest model_*.pt)')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--timesteps', type=int, default=100,
                   help='DDIM sampling steps')
    p.add_argument('--eta', type=float, default=0.0)
    p.add_argument('--gen_limit', type=int, default=0,
                   help='Cap generated private rows for smoke tests (0 = all)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cuda_device', type=str, default='0')
    return p.parse_args()


cli = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = cli.cuda_device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cli.seed)
np.random.seed(cli.seed)

temp = {k: SimpleNamespace(**v) for k, v in args.items()}
config = SimpleNamespace(**temp)

N_STEPS = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end,
                      N_STEPS).to(device)


# ---- model definitions: MUST match 9-C exactly (state-dict compatible) ----
class RouteEncoder(nn.Module):
    def __init__(self, vocab, out_dim, d_model=128, n_layers=2, n_heads=4, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, d_model, padding_idx=vocab)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, route):
        pad_mask = route == self.emb.padding_idx
        h = self.emb(route) + self.pos[:, :route.size(1)]
        h = self.enc(h, src_key_padding_mask=pad_mask)
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        denom = (~pad_mask).sum(dim=1, keepdim=True).clamp_min(1)
        return self.out(h.sum(dim=1) / denom)


class RouteGuideUNet(nn.Module):
    def __init__(self, config, route_vocab):
        super().__init__()
        self.ch = config.model.ch * 4
        self.guidance_scale = config.model.guidance_scale
        self.unet = Model(config)
        self.guide_emb = WideAndDeep(self.ch)
        self.place_emb = WideAndDeep(self.ch)
        self.route_enc = RouteEncoder(route_vocab, self.ch)

    def forward(self, x, t, attr, route):
        r = self.route_enc(route)
        cond = self.unet(x, t, self.guide_emb(attr) + r)
        place = torch.zeros(attr.shape, device=attr.device)
        uncond = self.unet(x, t, self.place_emb(place) + r)
        return cond + self.guidance_scale * (cond - uncond)


# ---- main loop over trained noise levels (5-A style) ----
meta = json.load(open(os.path.join(cli.osm_dir, 'routes_meta.json')))
VOCAB = int(meta['n_edges'])
print(f"Route vocab: {VOCAB} OSM edges")

head_global = np.load(cli.head_path, allow_pickle=True).astype(np.float32)
priv_idx = np.where(head_global[:, -1] > 1e-8)[0]
if cli.gen_limit > 0:
    priv_idx = priv_idx[:cli.gen_limit]
print(f"Private rows to generate: {len(priv_idx)}")

noise_dirs = sorted(d for d in os.listdir(cli.model_root) if d.startswith('noise_'))
if cli.noise_levels.strip():
    wanted = {s.strip() for s in cli.noise_levels.split(',') if s.strip()}
    noise_dirs = [d for d in noise_dirs if d.split('noise_')[1] in wanted]
assert noise_dirs, f"No trained noise_* dirs found under {cli.model_root}"
print("Found trained levels:", noise_dirs)

os.makedirs(cli.result_root, exist_ok=True)

skip = N_STEPS // cli.timesteps
seq = list(range(0, N_STEPS, skip))
seq_next = [-1] + seq[:-1]

for noise_dir in noise_dirs:
    nl = noise_dir.split('noise_')[1]
    model_dir = os.path.join(cli.model_root, noise_dir)
    if cli.ckpt_name:
        ckpt = os.path.join(model_dir, cli.ckpt_name)
    else:
        cands = sorted((f for f in os.listdir(model_dir)
                        if f.startswith('model_') and f.endswith('.pt')),
                       key=lambda f: int(f.split('_')[1].split('.')[0]))
        assert cands, f"No model_*.pt in {model_dir}"
        ckpt = os.path.join(model_dir, cands[-1])
    assert os.path.exists(ckpt), f"Missing checkpoint: {ckpt}"
    print(f"\n===== ControlTraj gen | noise {nl} | ckpt {os.path.basename(ckpt)} =====")

    routes = np.load(os.path.join(cli.osm_dir, f'routes_noise_{nl}.npy'))
    assert len(routes) == len(head_global), "route cache rows must align with the head"

    model = RouteGuideUNet(config, VOCAB).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    Gen_traj = []
    attr_priv = torch.from_numpy(head_global[priv_idx, :config.data.head_dim])
    route_priv = torch.from_numpy(routes[priv_idx])
    gen_dl = DataLoader(TensorDataset(attr_priv, route_priv),
                        batch_size=cli.batch_size, shuffle=False)

    for attr, route in tqdm(gen_dl, desc=f"gen ({nl})"):
        attr, route = attr.to(device), route.to(device)
        B = attr.size(0)
        x = torch.randn(B, 2, config.data.traj_length, device=device)
        with torch.no_grad():
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full((B,), i, device=device)
                next_t = torch.full((B,), j, device=device)
                pred = model(x, t, attr, route)
                x = p_xt(x, pred, t, next_t, beta, cli.eta)
        trajs = x.cpu().numpy()[:, :2, :]
        for b in range(B):
            Gen_traj.append(trajs[b].T.astype(float))

    save_path = os.path.join(cli.result_root, f'Gen_traj_noise_{nl}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(Gen_traj, f)
    print(f"Saved {len(Gen_traj)} trajectories to {save_path}")

print("\n🎉 All levels generated (ControlTraj-style).")
