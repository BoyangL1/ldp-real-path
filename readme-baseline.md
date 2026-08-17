# Baselines: LSTM-TrajGAN / PrivTrace / ControlTraj

All baselines see the exact same input as LDP-DiffTraj training
(`noise_sweep/noise_{nl}/`), one model per noise level. Each 9-* script
trains AND generates; the matching 10-* script regenerates from saved
checkpoints without retraining (`--seed`; `--n_samples` for 10-B,
`--timesteps` for 10-C).

| Train + generate | Regenerate | Method |
| --- | --- | --- |
| `9-A-baseline_lstm_trajgan.py` | `10-A-gen_lstm_trajgan.py` | LSTM-TrajGAN (GAN) |
| `9-B-baseline_privtrace.py` | `10-B-gen_privtrace.py` | PrivTrace (grid + Markov) |
| `9-C-baseline_controltraj.py` | `10-C-gen_controltraj.py` | ControlTraj (OSM-road-guided diffusion) |

Prereqs: main pipeline steps 0-3 done for the city; `pip install osmnx scipy`.
9-C downloads the OSM network on first run (cached to `data/osm_routes/<city>/`;
pre-upload that dir if the server is offline).

## Inputs / outputs (per city)

```
in :  data/traj_privacy/<city>/noise_sweep/noise_{nl}/{traj.npy, traj_features.npy}
      data/traj_privacy/<city>/{trajectory_features.npy, trajectory_tensor_stats.npz}
osm:  data/osm_routes/<city>/            (9-C cache: graph + per-level routes)
model: Baselines_<city>/<method>/noise_{nl}/       (G.pt | markov.pkl | model_*.pt)
out : Baseline_result_<city>/<method>/Gen_traj_noise_{nl}.pkl
```

Outputs match the 5-A/5-B pkl format and plug straight into the step-6
evaluation.

## Training commands (city = nagoya)

Every script accepts `--noise_levels 0.00,0.50` for splitting work across
processes. For another city, replace `nagoya` in every path and use that
city's bbox (must match 0-grid_privacy.py).

### LSTM-TrajGAN

```bash
python 9-A-baseline_lstm_trajgan.py \
    --sweep_root data/traj_privacy/nagoya/noise_sweep \
    --head_path  data/traj_privacy/nagoya/trajectory_features.npy \
    --model_root  Baselines_nagoya/lstm_trajgan \
    --result_root Baseline_result_nagoya/lstm_trajgan \
    --epochs 200 --cuda_device 0
```

### PrivTrace

```bash
python 9-B-baseline_privtrace.py \
    --sweep_root data/traj_privacy/nagoya/noise_sweep \
    --head_path  data/traj_privacy/nagoya/trajectory_features.npy \
    --stats_path data/traj_privacy/nagoya/trajectory_tensor_stats.npz \
    --model_root  Baselines_nagoya/privtrace \
    --result_root Baseline_result_nagoya/privtrace
```

### ControlTraj

```bash
python 9-C-baseline_controltraj.py \
    --sweep_root data/traj_privacy/nagoya/noise_sweep \
    --head_path  data/traj_privacy/nagoya/trajectory_features.npy \
    --stats_path data/traj_privacy/nagoya/trajectory_tensor_stats.npz \
    --min_lon 136.852 --max_lon 136.962 --min_lat 35.126 --max_lat 35.216 \
    --osm_dir data/osm_routes/nagoya \
    --model_root  Baselines_nagoya/controltraj \
    --result_root Baseline_result_nagoya/controltraj \
    --n_epochs 200 --cuda_device 0
```

### Evaluate

```bash
for B in lstm_trajgan privtrace controltraj; do
    python 6-eval_metrics_iterative.py \
        --feature_file   data/traj_privacy/nagoya/trajectory_features.npy \
        --real_traj_file data/traj_privacy/nagoya/noise_sweep/noise_0.00/traj.npy \
        --gen_dir        Baseline_result_nagoya/${B} \
        --out_csv        Baseline_result_nagoya/${B}/eval_summary.csv
done
```

## Notes

1. PrivTrace samples unconditionally: outputs are NOT row-aligned -
   distribution-level metrics only. 9-A / 9-C outputs are row-aligned.
2. PrivTrace internal DP is off by default (`--eps 0`); privacy comes from
   the LDP input.
3. Model classes in 10-A / 10-C must stay identical to 9-A / 9-C
   (state-dict compatibility).
4. Per-method deviations from the original papers are documented in each
   9-* script's docstring.

## Other cities

Same commands with `nagoya` replaced by the city name in every path
(`--sweep_root/--head_path/--stats_path/--osm_dir/--model_root/--result_root`).
Only 9-C additionally needs the city bbox (same values as run_one_city.sh):

| City | `--min_lon` | `--max_lon` | `--min_lat` | `--max_lat` |
| --- | --- | --- | --- | --- |
| tokyo | 139.698 | 139.808 | 35.640 | 35.730 |
| osaka | 135.448 | 135.557 | 34.649 | 34.739 |
| sapporo | 141.295 | 141.418 | 43.016 | 43.106 |

e.g. ControlTraj on tokyo:

```bash
python 9-C-baseline_controltraj.py \
    --sweep_root data/traj_privacy/tokyo/noise_sweep \
    --head_path  data/traj_privacy/tokyo/trajectory_features.npy \
    --stats_path data/traj_privacy/tokyo/trajectory_tensor_stats.npz \
    --min_lon 139.698 --max_lon 139.808 --min_lat 35.640 --max_lat 35.730 \
    --osm_dir data/osm_routes/tokyo \
    --model_root  Baselines_tokyo/controltraj \
    --result_root Baseline_result_tokyo/controltraj \
    --n_epochs 200 --cuda_device 0
```
