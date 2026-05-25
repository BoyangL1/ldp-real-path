# ldp-real-path

POI-aware LDP/DP trajectory diffusion for 4 Japanese cities.

| city    | N lat  | S lat  | W lon   | E lon   |
|---------|-------:|-------:|--------:|--------:|
| tokyo   | 35.730 | 35.640 | 139.698 | 139.808 |
| osaka   | 34.739 | 34.649 | 135.448 | 135.557 |
| nagoya  | 35.216 | 35.126 | 136.852 | 136.962 |
| sapporo | 43.106 | 43.016 | 141.295 | 141.418 |

Pipeline: `A`→`B` (one-off preprocessing) → `0`→`6` (per city) → `7` (cross-city plot).

To run the full per-city pipeline in one shot, use
[`run_one_city.sh`](run_one_city.sh) — see the bottom of this file.

---

## A — Extract per-city POIs (one-off)

```bash
python A_poi_split.py --src /mnt/data/poi/shop_info_genre.tsv \
                     --out-dir data/poi_4cities
```

## B — Extract per-city trajectories, 20231001–20231007 (one-off)

```bash
python B-split-traj.py --src-dir /mnt/data/haas \
                     --start 20231001 --end 20231007 \
                     --out-dir data/traj_4cities
```

---

## 0 — Grid privacy heatmap (bbox is the only per-city parameter)

```bash
python 0-grid_privacy.py --tsv data/poi_4cities/poi_tokyo_loco.tsv \
    --out-dir data/privacy_outputs/tokyo \
    --min-lon 139.698 --max-lon 139.808 --min-lat 35.640 --max-lat 35.730

python 0-grid_privacy.py --tsv data/poi_4cities/poi_osaka_loco.tsv \
    --out-dir data/privacy_outputs/osaka \
    --min-lon 135.448 --max-lon 135.557 --min-lat 34.649 --max-lat 34.739

python 0-grid_privacy.py --tsv data/poi_4cities/poi_nagoya_loco.tsv \
    --out-dir data/privacy_outputs/nagoya \
    --min-lon 136.852 --max-lon 136.962 --min-lat 35.126 --max-lat 35.216

python 0-grid_privacy.py --tsv data/poi_4cities/poi_sapporo_loco.tsv \
    --out-dir data/privacy_outputs/sapporo \
    --min-lon 141.295 --max-lon 141.418 --min-lat 43.016 --max-lat 43.106
```

## 1 — Split trajectories into per-user/day

```bash
python 1-split-traj.py --input data/traj_4cities/sim_tokyo_20231001_20231007.csv.gz
python 1-split-traj.py --input data/traj_4cities/sim_osaka_20231001_20231007.csv.gz
python 1-split-traj.py --input data/traj_4cities/sim_nagoya_20231001_20231007.csv.gz
python 1-split-traj.py --input data/traj_4cities/sim_sapporo_20231001_20231007.csv.gz
```

## 2 — Visualize trajectories

```bash
python 2-traj-visual.py --input data/traj_4cities/sim_tokyo_20231001_20231007_traj_compact.csv.gz   --output figs/compact_traj_tokyo.png
python 2-traj-visual.py --input data/traj_4cities/sim_osaka_20231001_20231007_traj_compact.csv.gz   --output figs/compact_traj_osaka.png
python 2-traj-visual.py --input data/traj_4cities/sim_nagoya_20231001_20231007_traj_compact.csv.gz  --output figs/compact_traj_nagoya.png
python 2-traj-visual.py --input data/traj_4cities/sim_sapporo_20231001_20231007_traj_compact.csv.gz --output figs/compact_traj_sapporo.png
```

## 3 — Snap to grid + noise sweep

```bash
python 3-traj-privacy-compact.py --city tokyo \
    --compact-csv data/traj_4cities/sim_tokyo_20231001_20231007_traj_compact.csv.gz \
    --grid-meta   data/privacy_outputs/tokyo/grid_meta.json \
    --privacy-csv data/privacy_outputs/tokyo/grid_privacy_scores.csv \
    --output-root data/traj_privacy/tokyo

python 3-traj-privacy-compact.py --city osaka \
    --compact-csv data/traj_4cities/sim_osaka_20231001_20231007_traj_compact.csv.gz \
    --grid-meta   data/privacy_outputs/osaka/grid_meta.json \
    --privacy-csv data/privacy_outputs/osaka/grid_privacy_scores.csv \
    --output-root data/traj_privacy/osaka

python 3-traj-privacy-compact.py --city nagoya \
    --compact-csv data/traj_4cities/sim_nagoya_20231001_20231007_traj_compact.csv.gz \
    --grid-meta   data/privacy_outputs/nagoya/grid_meta.json \
    --privacy-csv data/privacy_outputs/nagoya/grid_privacy_scores.csv \
    --output-root data/traj_privacy/nagoya

python 3-traj-privacy-compact.py --city sapporo \
    --compact-csv data/traj_4cities/sim_sapporo_20231001_20231007_traj_compact.csv.gz \
    --grid-meta   data/privacy_outputs/sapporo/grid_meta.json \
    --privacy-csv data/privacy_outputs/sapporo/grid_privacy_scores.csv \
    --output-root data/traj_privacy/sapporo
```

## 4 — Train LDP + DP diffusion (long-running)

```bash
python 4-A-main-ldptraj.py --dataset tokyo   && python 4-B-main-dptraj.py --dataset tokyo
python 4-A-main-ldptraj.py --dataset osaka   && python 4-B-main-dptraj.py --dataset osaka
python 4-A-main-ldptraj.py --dataset nagoya  && python 4-B-main-dptraj.py --dataset nagoya
python 4-A-main-ldptraj.py --dataset sapporo && python 4-B-main-dptraj.py --dataset sapporo
```

## 5 — Generate trajectories from checkpoints

```bash
# LDP
python 5-A-traj_gen_iterative_ldp.py --head_path data/traj_privacy/tokyo/trajectory_features.npy   --root ./LDP-DiffTraj_tokyo   --result_root ./LDP_result_tokyo   --noise_prefix tokyo_noise_
python 5-A-traj_gen_iterative_ldp.py --head_path data/traj_privacy/osaka/trajectory_features.npy   --root ./LDP-DiffTraj_osaka   --result_root ./LDP_result_osaka   --noise_prefix osaka_noise_
python 5-A-traj_gen_iterative_ldp.py --head_path data/traj_privacy/nagoya/trajectory_features.npy  --root ./LDP-DiffTraj_nagoya  --result_root ./LDP_result_nagoya  --noise_prefix nagoya_noise_
python 5-A-traj_gen_iterative_ldp.py --head_path data/traj_privacy/sapporo/trajectory_features.npy --root ./LDP-DiffTraj_sapporo --result_root ./LDP_result_sapporo --noise_prefix sapporo_noise_

# DP
python 5-B-traj_gen_iterative_dp.py  --head_path data/traj_privacy/tokyo/trajectory_features.npy   --root ./DPTraj_tokyo   --result_root ./DP_result_tokyo   --noise_prefix tokyo_noise_
python 5-B-traj_gen_iterative_dp.py  --head_path data/traj_privacy/osaka/trajectory_features.npy   --root ./DPTraj_osaka   --result_root ./DP_result_osaka   --noise_prefix osaka_noise_
python 5-B-traj_gen_iterative_dp.py  --head_path data/traj_privacy/nagoya/trajectory_features.npy  --root ./DPTraj_nagoya  --result_root ./DP_result_nagoya  --noise_prefix nagoya_noise_
python 5-B-traj_gen_iterative_dp.py  --head_path data/traj_privacy/sapporo/trajectory_features.npy --root ./DPTraj_sapporo --result_root ./DP_result_sapporo --noise_prefix sapporo_noise_
```

## 6 — Evaluate metrics (LDP + DP per city)

```bash
for city in tokyo osaka nagoya sapporo; do
    python 6-eval_metrics_iterative.py \
        --feature_file   data/traj_privacy/${city}/trajectory_features.npy \
        --real_traj_file data/traj_privacy/${city}/noise_sweep/noise_0.00/traj.npy \
        --gen_dir        ./LDP_result_${city}
    python 6-eval_metrics_iterative.py \
        --feature_file   data/traj_privacy/${city}/trajectory_features.npy \
        --real_traj_file data/traj_privacy/${city}/noise_sweep/noise_0.00/traj.npy \
        --gen_dir        ./DP_result_${city}
done
```

## 7 — Cross-city comparison plot

```bash
python 7-plot_metrics.py \
    --csv ./LDP_result_tokyo/metrics_summary.csv \
          ./LDP_result_osaka/metrics_summary.csv \
          ./LDP_result_nagoya/metrics_summary.csv \
          ./LDP_result_sapporo/metrics_summary.csv \
          ./DP_result_tokyo/metrics_summary.csv \
          ./DP_result_osaka/metrics_summary.csv \
          ./DP_result_nagoya/metrics_summary.csv \
          ./DP_result_sapporo/metrics_summary.csv \
    --label LDP-tokyo LDP-osaka LDP-nagoya LDP-sapporo \
            DP-tokyo  DP-osaka  DP-nagoya  DP-sapporo \
    --out figs/metrics_4cities.png
```

---

## Shortcut: full pipeline for one city

```bash
bash run_one_city.sh tokyo   35.730 35.640 139.698 139.808
bash run_one_city.sh osaka   34.739 34.649 135.448 135.557
bash run_one_city.sh nagoya  35.216 35.126 136.852 136.962
bash run_one_city.sh sapporo 43.106 43.016 141.295 141.418
```

Each invocation runs steps 0–6 for that city. Run step 7 once after all 4 finish.
