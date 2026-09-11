#!/usr/bin/env bash
# Run preprocessing -> LDP training -> uniform adaptive generation -> evaluation.
#
# Usage:
#   bash run_one_city.sh <city> <north_lat> <south_lat> <west_lon> <east_lon>
#
# Example (all 4 cities):
#   bash run_one_city.sh tokyo   35.730 35.640 139.698 139.808
#   bash run_one_city.sh osaka   34.739 34.649 135.448 135.557
#   bash run_one_city.sh nagoya  35.216 35.126 136.852 136.962
#   bash run_one_city.sh sapporo 43.106 43.016 141.295 141.418
#
# After all 4 cities finish, run 7-plot_metrics.py once to overlay them
# (see the bottom of this file for the command).
#
# Prereqs (run once, not per city):
#   python A_poi_split.py
#   python B-split-traj.py --start 20231001 --end 20231007

set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <city> <north_lat> <south_lat> <west_lon> <east_lon>" >&2
    exit 1
fi

CITY="$1"
N="$2"
S="$3"
W="$4"
E="$5"
DATE_TAG="20231001_20231007"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRAJ_DIR="data/traj_4cities"
POI_TSV="data/poi_4cities/poi_${CITY}_loco.tsv"
TRAJ_CSV="${TRAJ_DIR}/sim_${CITY}_${DATE_TAG}.csv.gz"
COMPACT_CSV="${TRAJ_DIR}/sim_${CITY}_${DATE_TAG}_traj_compact.csv.gz"
PRIV_OUT="data/privacy_outputs/${CITY}"
TRAJ_PRIV_OUT="data/traj_privacy/${CITY}"

echo "================================================================"
echo "City: ${CITY}   bbox: N=${N} S=${S} W=${W} E=${E}"
echo "POI:        ${POI_TSV}"
echo "Trajectory: ${TRAJ_CSV}"
echo "================================================================"

# Step 0 — POI grid privacy (bbox is city-specific).
python 0-grid_privacy.py \
    --tsv "${POI_TSV}" \
    --out-dir "${PRIV_OUT}" \
    --min-lon "${W}" --max-lon "${E}" \
    --min-lat "${S}" --max-lat "${N}"

# Step 1 — split full-week CSV into per-user/day trajectories.
# Optionally cap whole trajectories (e.g. MAX_TRAJ=100000 for large cities like tokyo)
# so step 3 + training stay fast. Unset/empty -> no cap.
python 1-split-traj.py --input "${TRAJ_CSV}" ${MAX_TRAJ:+--max-traj "${MAX_TRAJ}"}

# Step 2 — quick visual sanity check.
mkdir -p figs
python 2-traj-visual.py \
    --input  "${COMPACT_CSV}" \
    --output "figs/compact_traj_${CITY}.png"

# Step 3 — snap trajectories onto privacy grid + noise sweep.
python 3-traj-privacy-compact.py \
    --compact-csv "${COMPACT_CSV}" \
    --grid-meta   "${PRIV_OUT}/grid_meta.json" \
    --privacy-csv "${PRIV_OUT}/grid_privacy_scores.csv" \
    --city        "${CITY}" \
    --output-root "${TRAJ_PRIV_OUT}"

# Step 4: train LDP; generation/evaluation can also be run separately.
CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}" python 4-A-main-ldptraj.py --dataset "${CITY}"

# Uniform adaptive generation, metrics, and city plot.
bash run_adaptive_4cities.sh "${CITY}"
