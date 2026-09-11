#!/usr/bin/env bash
# Existing data/checkpoints -> uniform adaptive generation -> metrics -> plot.
# Optional positional cities; without arguments runs all four.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python}"
CITIES=("$@")
if [ "${#CITIES[@]}" -eq 0 ]; then CITIES=(tokyo osaka nagoya sapporo); fi
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-adaptive_snr_k0.1_id4_seed${SEED}}"
NOISE_LEVELS="${NOISE_LEVELS:-0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-200}"
STAGE="${STAGE:-all}"
case "$STAGE" in all|generate|eval) ;; *) echo 'STAGE must be all, generate, or eval' >&2; exit 2;; esac
execute() {
    if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
CSV_FILES=()
LABELS=()
for CITY in "${CITIES[@]}"; do
    case "$CITY" in tokyo|osaka|nagoya|sapporo) ;; *) echo "Unknown city: $CITY" >&2; exit 2;; esac
    DATA="data/traj_privacy/${CITY}"
    RESULT="LDP_result_${CITY}_adaptive"
    GEN_DIR="${RESULT}/${RUN_TAG}/guide_1"
    if [ "$STAGE" != eval ]; then
        execute "$PYTHON_BIN" 5-A2-traj_gen_retrieval_guided_ldp.py \
            --traj_path "${DATA}/noise_sweep/noise_0.00/traj.npy" \
            --head_path "${DATA}/trajectory_features.npy" --train_head_dir "${DATA}/noise_sweep" \
            --root "LDP-DiffTraj_${CITY}" --noise_prefix "${CITY}_noise_" \
            --result_root "$RESULT" --run_tag "$RUN_TAG" --noise_levels "$NOISE_LEVELS" \
            --cuda_device "$CUDA_DEVICE" --batch_size "$BATCH_SIZE" --seed "$SEED" \
            --timesteps 100 --eta 0 --adaptive --adapt_norm snr --adapt_key ti \
            --adapt_snr_scale 0.1 --adapt_gamma 1 --lam_ratio_min 0 \
            --guide_lambdas 1.0 --head_id_weight 4
    fi
    if [ "$STAGE" != generate ]; then
        execute "$PYTHON_BIN" 6-eval_metrics_iterative.py \
            --feature_file "${DATA}/trajectory_features.npy" \
            --real_traj_file "${DATA}/noise_sweep/noise_0.00/traj.npy" --gen_dir "$GEN_DIR"
    fi
    CSV_FILES+=("${GEN_DIR}/metrics_summary.csv")
    LABELS+=("${CITY}-adaptive")
done
if [ "$STAGE" != generate ]; then
    execute "$PYTHON_BIN" 7-plot_metrics.py --csv "${CSV_FILES[@]}" --label "${LABELS[@]}" \
        --out "figs/${RUN_TAG}_metrics.png" --title 'Uniform adaptive guidance: city comparison'
fi
