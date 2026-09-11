# Adaptive generation and evaluation

Run from `ldp-real-path` with preprocessed data, trained models, and a CUDA-enabled Python environment. Generation uses adaptive SNR guidance at all available noise levels. Evaluation saves SSIM, JSD, and Top-K F1 in `metrics_summary.csv`.

## Tokyo

Generate:

```bash
python 5-A2-traj_gen_retrieval_guided_ldp.py \
  --traj_path data/traj_privacy/tokyo/noise_sweep/noise_0.00/traj.npy \
  --head_path data/traj_privacy/tokyo/trajectory_features.npy \
  --train_head_dir data/traj_privacy/tokyo/noise_sweep \
  --root LDP-DiffTraj_tokyo --noise_prefix tokyo_noise_ \
  --result_root LDP_result_tokyo_adaptive --run_tag adaptive_snr_k0.1_id4_seed0 \
  --cuda_device 0 --adaptive --adapt_norm snr --adapt_snr_scale 0.1 --head_id_weight 4
```

Evaluate:

```bash
python 6-eval_metrics_iterative.py \
  --feature_file data/traj_privacy/tokyo/trajectory_features.npy \
  --real_traj_file data/traj_privacy/tokyo/noise_sweep/noise_0.00/traj.npy \
  --gen_dir LDP_result_tokyo_adaptive/adaptive_snr_k0.1_id4_seed0/guide_1
```

## Osaka

Generate:

```bash
python 5-A2-traj_gen_retrieval_guided_ldp.py \
  --traj_path data/traj_privacy/osaka/noise_sweep/noise_0.00/traj.npy \
  --head_path data/traj_privacy/osaka/trajectory_features.npy \
  --train_head_dir data/traj_privacy/osaka/noise_sweep \
  --root LDP-DiffTraj_osaka --noise_prefix osaka_noise_ \
  --result_root LDP_result_osaka_adaptive --run_tag adaptive_snr_k0.1_id4_seed0 \
  --cuda_device 0 --adaptive --adapt_norm snr --adapt_snr_scale 0.1 --head_id_weight 4
```

Evaluate:

```bash
python 6-eval_metrics_iterative.py \
  --feature_file data/traj_privacy/osaka/trajectory_features.npy \
  --real_traj_file data/traj_privacy/osaka/noise_sweep/noise_0.00/traj.npy \
  --gen_dir LDP_result_osaka_adaptive/adaptive_snr_k0.1_id4_seed0/guide_1
```

## Nagoya

Generate:

```bash
python 5-A2-traj_gen_retrieval_guided_ldp.py \
  --traj_path data/traj_privacy/nagoya/noise_sweep/noise_0.00/traj.npy \
  --head_path data/traj_privacy/nagoya/trajectory_features.npy \
  --train_head_dir data/traj_privacy/nagoya/noise_sweep \
  --root LDP-DiffTraj_nagoya --noise_prefix nagoya_noise_ \
  --result_root LDP_result_nagoya_adaptive --run_tag adaptive_snr_k0.1_id4_seed0 \
  --cuda_device 0 --adaptive --adapt_norm snr --adapt_snr_scale 0.1 --head_id_weight 4
```

Evaluate:

```bash
python 6-eval_metrics_iterative.py \
  --feature_file data/traj_privacy/nagoya/trajectory_features.npy \
  --real_traj_file data/traj_privacy/nagoya/noise_sweep/noise_0.00/traj.npy \
  --gen_dir LDP_result_nagoya_adaptive/adaptive_snr_k0.1_id4_seed0/guide_1
```

## Sapporo

Generate:

```bash
python 5-A2-traj_gen_retrieval_guided_ldp.py \
  --traj_path data/traj_privacy/sapporo/noise_sweep/noise_0.00/traj.npy \
  --head_path data/traj_privacy/sapporo/trajectory_features.npy \
  --train_head_dir data/traj_privacy/sapporo/noise_sweep \
  --root LDP-DiffTraj_sapporo --noise_prefix sapporo_noise_ \
  --result_root LDP_result_sapporo_adaptive --run_tag adaptive_snr_k0.1_id4_seed0 \
  --cuda_device 0 --adaptive --adapt_norm snr --adapt_snr_scale 0.1 --head_id_weight 4
```

Evaluate:

```bash
python 6-eval_metrics_iterative.py \
  --feature_file data/traj_privacy/sapporo/trajectory_features.npy \
  --real_traj_file data/traj_privacy/sapporo/noise_sweep/noise_0.00/traj.npy \
  --gen_dir LDP_result_sapporo_adaptive/adaptive_snr_k0.1_id4_seed0/guide_1
```
