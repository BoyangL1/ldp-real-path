"""
Plot evaluation metrics produced by 6-eval_metrics_iterative.py
into a 2x3 grid: SSIM-OD-avg, SSIM-occupancy, Top-K F1,
                 length JSD, high-sens OD JSD, high-sens occupancy JSD.

Supports overlaying multiple CSVs (e.g. LDP vs DP vs baseline noise_sweep).
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


METRICS = [
    ("ssim_od_avg",            "SSIM (OD avg)",                "↑"),
    ("ssim_occupancy",         "SSIM (occupancy)",             "↑"),
    ("topk_f1",                "Top-K F1",                     "↑"),
    ("length_jsd",             "Length JSD",                   "↓"),
    ("od_jsd_high_sens",       "OD JSD (high sensitivity)",    "↓"),
    ("occupancy_jsd_high_sens","Occupancy JSD (high sensitivity)", "↓"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Plot metrics from one or more summary CSVs")
    p.add_argument('--csv', type=str, nargs='+',
                   default=['./LDP_result_nagoya/metrics_summary.csv'],
                   help='One or more metrics_summary.csv files to overlay')
    p.add_argument('--label', type=str, nargs='+', default=None,
                   help='Legend label per CSV (defaults to parent dir name)')
    p.add_argument('--out', type=str, default='figs/metrics_summary.png',
                   help='Output image path (saved under figs/ by default)')
    p.add_argument('--title', type=str, default='Trajectory generation metrics vs. noise')
    return p.parse_args()


def main():
    cli = parse_args()

    if cli.label is None:
        labels = [os.path.basename(os.path.dirname(os.path.abspath(c))) or c for c in cli.csv]
    else:
        if len(cli.label) != len(cli.csv):
            raise ValueError(f"--label count ({len(cli.label)}) must match --csv count ({len(cli.csv)})")
        labels = cli.label

    dfs = []
    for path, label in zip(cli.csv, labels):
        df = pd.read_csv(path).sort_values('noise').reset_index(drop=True)
        dfs.append((label, df))

    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()

    for ax, (col, name, arrow) in zip(axes, METRICS):
        for label, df in dfs:
            if col not in df.columns:
                continue
            ax.plot(df['noise'], df[col], marker='o', linewidth=1.6, label=label)
        ax.set_title(f"{name}  ({arrow})")
        ax.set_xlabel('noise level')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc='upper center',
                   ncol=len(handles), bbox_to_anchor=(0.5, 1.02), frameon=False)

    fig.suptitle(cli.title, y=1.06 if handles else 1.00)
    fig.tight_layout()
    fig.savefig(cli.out, dpi=160, bbox_inches='tight')
    print(f"Saved figure to: {cli.out}")


if __name__ == '__main__':
    main()
