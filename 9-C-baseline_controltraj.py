"""
Baseline 9-C: ControlTraj-style road-network-guided diffusion on the Nagoya
LDP noise sweep. SELF-CONTAINED: give it the bbox lon/lat, it downloads the
real OSM road network, snaps the noised trajectories to road segments, then
trains and generates per noise level.

Reference
---------
Zhu et al.: "ControlTraj: Controllable Trajectory Generation with
Topology-Constrained Diffusion Model" (KDD 2024). Official code:
https://github.com/Yasoz/ControlTraj (GeoUNet + RoadMAE road-segment
autoencoder conditioned on the route).

Pipeline of this script (one file, in order)
--------------------------------------------
  1. Download the OSM drive network for the given bbox (cached to graphml,
     so the server only needs internet on the first run).
  2. Per noise level: snap every point of the NOISED trajectories to the
     nearest road edge (nearest-edge map matching over a 25 m-sampled edge
     index) -> road-edge-id route sequences (cached to .npy).
  3. Train the route-conditioned diffusion model per noise level and
     generate row-aligned private trajectories.

Adaptations vs the original (documented for the paper)
------------------------------------------------------
  * RoadMAE -> a transformer route encoder trained END-TO-END with the
    diffusion loss (no separate masked pretraining). Its pooled embedding is
    added to the UNet timestep embedding (`extra_embed` hook) - the same
    place ControlTraj injects road embeddings; CFG runs on trip attributes
    as in Guide_UNet, the route conditions BOTH branches.
  * Map matching is nearest-edge snapping (not HMM); at 20 points / 100 m
    grid resolution this is the appropriate fidelity.
  * Training uses UNIFORM diffusion timesteps (vanilla DDPM) - none of our
    privacy-aligned timestep machinery.

Protocol (identical input to LDP-DiffTraj training)
---------------------------------------------------
Per noise level nl the model sees ONLY noise_sweep/noise_{nl}/traj.npy and
its traj_features.npy head. The road network is PUBLIC; routes are derived
from the NOISED trajectories (never from clean private data). At high noise
the snapped routes degrade honestly - that is the point of the baseline.

Output: <result_root>/Gen_traj_noise_{nl}.pkl - list of [Len,2] float arrays
(z-scored grid space), row-aligned with the private subset like 5-A.

Usage
-----
  python 9-C-baseline_controltraj.py                      # all noise levels
  python 9-C-baseline_controltraj.py --noise_levels 0.50 --n_epochs 2 \
      --max_train 512 --gen_limit 256 --timesteps 20      # smoke test
  # another city: pass --min_lon/--max_lon/--min_lat/--max_lat/--cell_m
  # matching the values used in 0-grid_privacy.py for that dataset.
"""
import argparse
import json
import math
import os
import pickle
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.config import args
from utils.Traj_UNet import Model, WideAndDeep
from utils.utils import q_xt_x0, p_xt


# =========================
# 0. CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="ControlTraj-style baseline: OSM download + route snap + training")
    # data
    p.add_argument('--sweep_root', type=str,
                   default='data/traj_privacy/nagoya/noise_sweep')
    p.add_argument('--head_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_features.npy')
    p.add_argument('--stats_path', type=str,
                   default='data/traj_privacy/nagoya/trajectory_tensor_stats.npz')
    p.add_argument('--model_root', type=str, default='./Baselines_nagoya/controltraj')
    p.add_argument('--result_root', type=str, default='./Baseline_result_nagoya/controltraj')
    p.add_argument('--noise_levels', type=str, default='',
                   help='Comma-separated subset, e.g. "0.00,0.50". Empty = all.')
    # bbox / grid geometry — MUST match 0-grid_privacy.py for the dataset
    p.add_argument('--min_lon', type=float, default=136.8516)
    p.add_argument('--max_lon', type=float, default=136.9616)
    p.add_argument('--min_lat', type=float, default=35.1365)
    p.add_argument('--max_lat', type=float, default=35.2265)
    p.add_argument('--cell_m', type=float, default=100.0)
    # OSM / routes
    p.add_argument('--osm_dir', type=str, default='data/osm_routes/nagoya',
                   help='Cache dir for the OSM graph and per-level route files')
    p.add_argument('--bbox_buffer', type=float, default=0.005,
                   help='Degrees of buffer around the bbox for the OSM download')
    p.add_argument('--edge_sample_m', type=float, default=25.0,
                   help='Sampling interval along road edges for the snap index')
    p.add_argument('--route_len', type=int, default=32)
    p.add_argument('--rebuild_routes', action='store_true',
                   help='Ignore cached route files and re-snap')
    # training / generation
    p.add_argument('--n_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--save_every', type=int, default=100)
    p.add_argument('--timesteps', type=int, default=100,
                   help='DDIM sampling steps at generation')
    p.add_argument('--eta', type=float, default=0.0)
    p.add_argument('--max_train', type=int, default=0,
                   help='Subsample training rows for smoke tests (0 = all)')
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
alpha = 1.0 - beta
alpha_bar = alpha.cumprod(dim=0)

# grid -> lon/lat mapping, exactly as 0-grid_privacy.py grid_spec()
CENTER_LAT = 0.5 * (cli.min_lat + cli.max_lat)
D_LAT = cli.cell_m / 111_000.0
D_LON = cli.cell_m / (111_000.0 * math.cos(math.radians(CENTER_LAT)))

# local equirectangular meters for snapping
MX = 111_320.0 * math.cos(math.radians(CENTER_LAT))
MY = 110_540.0


# =========================
# 1. OSM road network + snap index (built lazily, cached on disk)
# =========================
class RoadSnapper:
    def __init__(self):
        os.makedirs(cli.osm_dir, exist_ok=True)
        import osmnx as ox
        graph_path = os.path.join(cli.osm_dir, 'drive.graphml')
        if os.path.exists(graph_path):
            print("Loading cached OSM graph:", graph_path)
            G = ox.load_graphml(graph_path)
        else:
            bbox = (cli.min_lon - cli.bbox_buffer, cli.min_lat - cli.bbox_buffer,
                    cli.max_lon + cli.bbox_buffer, cli.max_lat + cli.bbox_buffer)
            print("Downloading OSM drive network for bbox", bbox)
            G = ox.graph_from_bbox(bbox=bbox, network_type='drive')
            ox.save_graphml(G, graph_path)
        edges = ox.convert.graph_to_gdfs(G, nodes=False).reset_index()
        self.n_edges = len(edges)
        print(f"OSM edges: {self.n_edges}")

        from scipy.spatial import cKDTree
        samples, sample_edge = [], []
        for eid, geom in enumerate(edges.geometry):
            coords = np.asarray(geom.coords)
            seg = np.stack([coords[:, 0] * MX, coords[:, 1] * MY], axis=1)
            d = np.linalg.norm(np.diff(seg, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(d)])
            n = max(int(cum[-1] // cli.edge_sample_m), 1) + 1
            u = np.linspace(0, cum[-1], n)
            samples.append(np.stack([np.interp(u, cum, seg[:, 0]),
                                     np.interp(u, cum, seg[:, 1])], axis=1))
            sample_edge.append(np.full(n, eid, dtype=np.int64))
        self.sample_edge = np.concatenate(sample_edge)
        self.tree = cKDTree(np.concatenate(samples))
        print(f"Snap index: {len(self.sample_edge)} sampled road points")

    def routes_for(self, traj_z, mean, std):
        """[N,L,2] z-scored -> [N,route_len] edge-id tokens (PAD = n_edges)."""
        g = traj_z.astype(np.float64) * std + mean                   # grid coords
        lon = cli.min_lon + g[..., 0] * D_LON
        lat = cli.min_lat + g[..., 1] * D_LAT
        pts = np.stack([lon.ravel() * MX, lat.ravel() * MY], axis=1)
        dist, idx = self.tree.query(pts, workers=-1)
        edge_ids = self.sample_edge[idx].reshape(len(traj_z), -1)

        routes = np.full((len(traj_z), cli.route_len), self.n_edges, dtype=np.int64)
        for i, row in enumerate(edge_ids):
            seq = [row[0]]
            for v in row[1:]:
                if v != seq[-1]:
                    seq.append(v)
            seq = seq[:cli.route_len]
            routes[i, :len(seq)] = seq
        print(f"  snap: mean {dist.mean():6.1f} m | p90 {np.quantile(dist, .9):6.1f} m "
              f"| mean route len {(routes != self.n_edges).sum(1).mean():5.1f}")
        return routes


_snapper = None


def get_snapper():
    global _snapper
    if _snapper is None:
        _snapper = RoadSnapper()
    return _snapper


def load_or_build_routes(nl, traj_z, mean, std):
    """Per-level route cache; needs OSM/network only when the cache is missing."""
    meta_path = os.path.join(cli.osm_dir, 'routes_meta.json')
    route_path = os.path.join(cli.osm_dir, f'routes_noise_{nl}.npy')
    if not cli.rebuild_routes and os.path.exists(route_path) and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta.get('route_len') == cli.route_len:
            return np.load(route_path), int(meta['n_edges'])
    snapper = get_snapper()
    routes = snapper.routes_for(traj_z, mean, std)
    np.save(route_path, routes)
    with open(meta_path, 'w') as f:
        json.dump({'n_edges': int(snapper.n_edges), 'route_len': cli.route_len,
                   'pad_id': int(snapper.n_edges)}, f, indent=2)
    return routes, snapper.n_edges


# =========================
# 2. Route encoder + guided UNet
# =========================
class RouteEncoder(nn.Module):
    """Transformer over road-edge route tokens -> pooled embedding (RoadMAE stand-in)."""

    def __init__(self, vocab, out_dim, d_model=128, n_layers=2, n_heads=4, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab + 1, d_model, padding_idx=vocab)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, route):                                        # [B, L] int64
        pad_mask = route == self.emb.padding_idx
        h = self.emb(route) + self.pos[:, :route.size(1)]
        h = self.enc(h, src_key_padding_mask=pad_mask)
        h = h.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        denom = (~pad_mask).sum(dim=1, keepdim=True).clamp_min(1)
        return self.out(h.sum(dim=1) / denom)                        # [B, out_dim]


class RouteGuideUNet(nn.Module):
    """Guide_UNet with an additional route embedding on both CFG branches."""

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


# =========================
# 3. Main loop over noise levels
# =========================
stats = np.load(cli.stats_path)
MEAN, STD = stats['mean'].reshape(1, 1, 2), stats['std'].reshape(1, 1, 2)

head_global = np.load(cli.head_path, allow_pickle=True).astype(np.float32)
priv_idx = np.where(head_global[:, -1] > 1e-8)[0]
if cli.gen_limit > 0:
    priv_idx = priv_idx[:cli.gen_limit]
print(f"Total rows: {len(head_global)}  |  private rows to generate: {len(priv_idx)}")

level_dirs = sorted(d for d in os.listdir(cli.sweep_root) if d.startswith('noise_'))
levels = [d.split('noise_')[1] for d in level_dirs]
if cli.noise_levels.strip():
    wanted = {s.strip() for s in cli.noise_levels.split(',') if s.strip()}
    levels = [nl for nl in levels if nl in wanted]
    assert levels, f"None of the requested noise levels {wanted} found under {cli.sweep_root}"
print("Processing noise levels:", levels)

os.makedirs(cli.result_root, exist_ok=True)

skip = N_STEPS // cli.timesteps
seq = list(range(0, N_STEPS, skip))
seq_next = [-1] + seq[:-1]

for nl in levels:
    print(f"\n===== ControlTraj-style | noise {nl} =====")
    traj = np.load(os.path.join(cli.sweep_root, f'noise_{nl}', 'traj.npy')).astype(np.float32)
    feats = np.load(os.path.join(cli.sweep_root, f'noise_{nl}', 'traj_features.npy')).astype(np.float32)
    assert len(traj) == len(head_global)

    routes, vocab = load_or_build_routes(nl, traj, MEAN, STD)
    assert len(routes) == len(traj)

    train_ids = np.arange(len(traj))
    if cli.max_train > 0 and cli.max_train < len(train_ids):
        train_ids = np.random.choice(train_ids, cli.max_train, replace=False)

    x_all = torch.from_numpy(np.swapaxes(traj, 1, 2))                # [N,2,L]
    ds = TensorDataset(x_all[train_ids],
                       torch.from_numpy(feats[train_ids, :config.data.head_dim]),
                       torch.from_numpy(routes[train_ids]))
    dl = DataLoader(ds, batch_size=cli.batch_size, shuffle=True, drop_last=True)

    model = RouteGuideUNet(config, vocab).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cli.lr)

    model_dir = os.path.join(cli.model_root, f'noise_{nl}')
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(1, cli.n_epochs + 1):
        losses = []
        for x0, attr, route in dl:
            x0, attr, route = x0.to(device), attr.to(device), route.to(device)
            t = torch.randint(0, N_STEPS, (len(x0),), device=device)
            xt, noise = q_xt_x0(x0, t, alpha_bar)
            pred = model(xt.float(), t, attr, route)
            loss = F.mse_loss(noise.float(), pred)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 1 or epoch == cli.n_epochs:
            print(f"  epoch {epoch:4d} | loss {np.mean(losses):.6f}")
        if epoch % cli.save_every == 0 or epoch == cli.n_epochs:
            torch.save(model.state_dict(), os.path.join(model_dir, f'model_{epoch}.pt'))

    # ---- generation: DDIM conditioned on private heads + noised-traj routes
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

print("\n🎉 ControlTraj-style baseline finished.")
