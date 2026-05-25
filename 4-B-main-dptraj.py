import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import argparse
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from utils.config import args
from utils.EMA import EMAHelper
from utils.Traj_UNet import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil
from torch.utils.data import Sampler

class PrivacySortBatchSampler(Sampler):
    """
    Sort by privacy score (e.g., head[:, -1]) and batch contiguous samples together.
    Batch-level shuffle to keep randomness (Global Shuffle, Local Sort).
    """
    def __init__(self, privacy_scores, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_samples = len(privacy_scores)

        if isinstance(privacy_scores, torch.Tensor):
            privacy_scores = privacy_scores.detach().cpu().numpy()

        self.sorted_indices = np.argsort(privacy_scores)

    def __iter__(self):
        batches = []
        for i in range(0, self.num_samples, self.batch_size):
            if i + self.batch_size > self.num_samples:
                if not self.drop_last:
                    batches.append(self.sorted_indices[i:])
            else:
                batches.append(self.sorted_indices[i:i + self.batch_size])

        np.random.shuffle(batches)  # shuffle batches, not samples inside a batch
        for b in batches:
            yield b.tolist()

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size

# This code part from https://github.com/sunlin-ai/diffusion_tutorial


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1)


def main(config, logger, exp_dir):

    # Modified to return the noise itself as well
    def q_xt_x0(x0, t):
        mean = gather(alpha_bar, t)**0.5 * x0
        var = 1 - gather(alpha_bar, t)
        eps = torch.randn_like(x0).to(x0.device)
        return mean + (var**0.5) * eps, eps  # also returns noise

    # Create the model
    unet = Guide_UNet(config).cuda()
    # print(unet)
    traj = np.load(config.data.traj_path,allow_pickle=True)
    head_full = np.load(config.data.head_path, allow_pickle=True)
    traj = np.swapaxes(traj, 1, 2)

    traj = torch.from_numpy(traj).float()
    head_full = torch.from_numpy(head_full).float()

    privacy_scores = head_full[:, -1].long()   # shape: [N]

    head_cond = head_full[:, :config.data.head_dim]  # shape: [N, head_dim]

    dataset = TensorDataset(traj, head_cond, privacy_scores)

    batch_sampler = PrivacySortBatchSampler(
        privacy_scores=privacy_scores,
        batch_size=config.training.batch_size,
        drop_last=False
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=8,
        pin_memory=True
    )
    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda()
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    base_lr = 1e-4  
    weight_decay = 1e-4 

    losses = []  # Store losses for later plotting
    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, 
        T_max=config.training.n_epochs, 
        eta_min=1e-6
    )
    
    # EMA
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(unet)
    else:
        ema_helper = None

    # new filefold for save model pt
    model_save = exp_dir / 'models' / (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1
    for epoch in range(1, config.training.n_epochs + 1):
        logger.info("<----Epoch-{}---->".format(epoch))
        for iter_idx, (trainx, head, priv) in enumerate(dataloader):
            if iter_idx % 1 == 0:
                logger.info(f"privacy[min,max]=({priv.min().item()},{priv.max().item()})")
            x0 = trainx.cuda()
            head = head.cuda()
            # t = torch.randint(low=0, high=n_steps,
            #                   size=(len(x0) // 2 + 1, )).cuda()
            # t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)]
            t = torch.randint(
                low=0,
                high=n_steps,
                size=(len(x0),),
                device=x0.device
            )
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t)
            # Run xt through the network to get its predictions
            pred_noise = unet(xt.float(), t, head)
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise)
            # Store the loss for later viewing
            losses.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
            if config.model.ema:
                ema_helper.update(unet)
            
            logger.info(f"Epoch {epoch}, Iter {iter_idx}, Loss: {loss.item():.6f}")
        scheduler.step()
        if (epoch) % 500 == 0:
            m_path = model_save / f"unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            m_path = exp_dir / 'results' / f"loss_{epoch}.npy"
            np.save(m_path, np.array(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name, corresponds to subfolder under data/traj_privacy/")
    cli_args = parser.parse_args()

    # Load base configuration
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    base_config = SimpleNamespace(**temp)

    root_dir = Path(__name__).resolve().parents[0]

    # noise sweep
    NOISE_LEVELS = [f"{i/10:.2f}" for i in range(0, 11)]  # 0.0 ~ 1.0

    for nl in NOISE_LEVELS:
        print(f"\n========== Training on noise_{nl} ==========")

        config = SimpleNamespace(**vars(base_config))
        config.data = SimpleNamespace(**vars(base_config.data))

        config.data.dataset = cli_args.dataset
        config.data.traj_path = f"data/traj_privacy/{config.data.dataset}/noise_sweep/noise_{nl}/traj.npy"
        config.data.head_path = f"data/traj_privacy/{config.data.dataset}/noise_sweep/noise_{nl}/traj_features.npy"

        result_name = (
            f"{config.data.dataset}_noise_{nl}_"
            f"steps={config.diffusion.num_diffusion_timesteps}_"
            f"len={config.data.traj_length}"
        )
        exp_dir = root_dir / f"DPTraj_{config.data.dataset}" / result_name

        for d in ["results", "models", "logs", "Files"]:
            os.makedirs(exp_dir / d, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")

        files_save = exp_dir / 'Files' / timestamp
        os.makedirs(files_save, exist_ok=True)
        shutil.copy('./utils/config.py', files_save)
        shutil.copy('./utils/Traj_UNet.py', files_save)

        logger = Logger(
            __name__,
            log_path=exp_dir / "logs" / f"{timestamp}.log",
            colorize=True,
        )
        log_info(config, logger)

        # 🚀 run training
        main(config, logger, exp_dir)