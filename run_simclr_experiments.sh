#!/usr/bin/env bash
# run_simclr_experiments.sh
# Runs SimCLR pretraining + linear‑probe + k‑NN eval for multiple augmentations.

set -euo pipefail

# -----------------------
# Configuration overrides
# -----------------------
# Default to a 1‑epoch smoke test; change these env vars for full runs:
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# GPUs to use
GPUS_PER_NODE=2

# Augmentation experiments (suffix of config filenames)
EXPERIMENTS=(baseline color blur gray all)

# Paths
CONFIG_DIR="configs"
RESULTS_DIR="results"
MODEL_NAME="simclr"

# Activate virtual environment if not already active
if [[ -z "${CONDA_DEFAULT_ENV-}" || "${CONDA_DEFAULT_ENV}" != "ssl_env" ]]; then
  echo "Activating ssl_env..."
  source ~/SSL/ssl_env/bin/activate
fi
echo "Using environment: $(conda info --envs | grep '*' || echo 'ssl_env')"

# -----------------------
# Main experiment loop
# -----------------------
for AUG in "${EXPERIMENTS[@]}"; do
  echo
  echo "========================================================="
  echo "   Experiment: ${MODEL_NAME} | Augmentation = ${AUG}   "
  echo "========================================================="

  CONFIG_FILE="${CONFIG_DIR}/${MODEL_NAME}_${AUG}.yaml"
  RUN_DIR="${RESULTS_DIR}/${MODEL_NAME}_${AUG}"
  CHECKPOINT_PATH="${RUN_DIR}/final_model.pth"

  # Verify config exists
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "⚠️  Config not found: $CONFIG_FILE  – skipping."
    continue
  fi

  mkdir -p "$RUN_DIR"

  # 1) Pretrain SimCLR
  echo "--> [1/3] Pretraining with ${AUG} augmentations (epochs=${EPOCHS}, batch_size=${BATCH_SIZE})..."
  torchrun --nproc_per_node=${GPUS_PER_NODE} simclr.py \
    --config "$CONFIG_FILE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    2>&1 | tee "${RUN_DIR}/pretrain.log"

  if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "❌  Checkpoint not found after pretraining. Skipping eval for ${AUG}."
    continue
  fi
  echo "✅  Pretraining complete. Checkpoint: $CHECKPOINT_PATH"

  # 2) Linear Probe
  echo "--> [2/3] Running Linear Probe evaluation..."
  python linear_probe.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_FILE" \
    2>&1 | tee "${RUN_DIR}/linear_probe.log"

  # 3) k-NN Evaluation
  echo "--> [3/3] Running k-NN evaluation..."
  python knn_eval.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_FILE" \
    2>&1 | tee "${RUN_DIR}/knn_eval.log"

  echo "✅  Completed experiment for: ${AUG}"
done

echo
echo "========================================================="
echo "   All experiments finished! Check '${RESULTS_DIR}/' for outputs."
echo "========================================================="
