import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from types import SimpleNamespace
from torch.utils.data import DataLoader
import pickle

from utils.Traj_UNet import *
from utils.config import args
from utils.utils import *

# =========================
# 0. CLI arguments
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Iterative trajectory generation for LDP-DiffTraj")
    parser.add_argument('--head_path', type=str,
                        default='data/traj_privacy/nagoya/trajectory_features.npy',
                        help='Path to trajectory_features.npy')
    parser.add_argument('--root', type=str,
                        default='./LDP-DiffTraj_nagoya',
                        help='Directory containing per-noise-level model subdirs')
    parser.add_argument('--result_root', type=str,
                        default='./LDP_result_nagoya',
                        help='Where to save generated trajectory pkls')
    parser.add_argument('--noise_prefix', type=str,
                        default='nagoya_noise_',
                        help='Prefix used to filter noise-level subdirs under --root')
    parser.add_argument('--ckpt_name', type=str,
                        default='unet_1000.pt',
                        help='Checkpoint filename inside each model timestamp dir')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--timesteps', type=int, default=100,
                        help='Number of DDIM sampling steps')
    parser.add_argument('--eta', type=float, default=0.0)
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='Value for CUDA_VISIBLE_DEVICES')
    return parser.parse_args()


cli_args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = cli_args.cuda_device

# =========================
# 1. Config
# =========================
temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)
config = SimpleNamespace(**temp)

device = "cuda"

# =========================
# 2. Diffusion settings
# =========================
n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(
    config.diffusion.beta_start,
    config.diffusion.beta_end,
    n_steps
).to(device)

eta = cli_args.eta
timesteps = cli_args.timesteps
skip = n_steps // timesteps
seq = list(range(0, n_steps, skip))
seq_next = [-1] + seq[:-1]

# =========================
# 3. Load head data
# =========================
batchsize = cli_args.batch_size
head = np.load(cli_args.head_path, allow_pickle=True)
head = torch.from_numpy(head).float()

mask = head[:, -1] != 0
head = head[mask]

privacy_budget = head[:, -1]
priv_idx = np.where(privacy_budget > 1e-8)[0]
head = head[priv_idx]

dataloader = DataLoader(
    head,
    batch_size=batchsize,
    shuffle=False,
    num_workers=4
)

print("Total heads:", len(head))

# =========================
# 4. Paths（🔥关键）
# =========================
ROOT = cli_args.root
RESULT_ROOT = cli_args.result_root
os.makedirs(RESULT_ROOT, exist_ok=True)

noise_dirs = sorted([
    d for d in os.listdir(ROOT)
    if d.startswith(cli_args.noise_prefix)
])

print("Found noise levels:", noise_dirs)

# =========================
# 5. Loop over noise levels
# =========================
for noise_dir in noise_dirs:
    print(f"\n===== Processing {noise_dir} =====")

    # 解析 noise level，用于命名
    noise_level = noise_dir.split("noise_")[1].split("_")[0]

    model_root = os.path.join(ROOT, noise_dir, "models")
    time_dirs = sorted(os.listdir(model_root))
    assert len(time_dirs) > 0, f"No model dirs in {model_root}"

    model_dir = os.path.join(model_root, time_dirs[-1])
    model_path = os.path.join(model_dir, cli_args.ckpt_name)
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    print("Loading model:", model_path)

    # =========================
    # 6. Load model
    # =========================
    unet = Guide_UNet(config).to(device)
    unet.load_state_dict(torch.load(model_path, map_location=device))
    unet.eval()

    Gen_traj = []

    # =========================
    # 7. Generate trajectories
    # =========================
    for head_batch in tqdm(dataloader, desc=f"Generating ({noise_level})"):
        head_batch = head_batch.to(device)
        B = head_batch.shape[0]

        x = torch.randn(B, 2, config.data.traj_length).to(device)
        ims = []

        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = torch.full((B,), i, device=device)
            next_t = torch.full((B,), j, device=device)

            with torch.no_grad():
                pred_noise = unet(x, t, head_batch)
                x = p_xt(x, pred_noise, t, next_t, beta, eta)

                if i % 10 == 0:
                    ims.append(x.cpu())

        trajs = ims[-1].numpy()[:, :2, :]

        for b in range(B):
            Gen_traj.append(trajs[b].T.astype(float))

    # =========================
    # 8. Save result（🔥统一目录）
    # =========================
    save_path = os.path.join(
        RESULT_ROOT,
        f"Gen_traj_noise_{noise_level}.pkl"
    )

    with open(save_path, "wb") as f:
        pickle.dump(Gen_traj, f)

    print(f"Saved trajectories to {save_path}")

print("\n🎉 All noise levels finished.")
