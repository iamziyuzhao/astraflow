#!/bin/bash
set -euo pipefail
# [3/3] Launch Trainer for model0 (Megatron TP2/EP2/DP3, TCP sender on rank 0)
#
# Runs on the 6-GPU H200 trainer node. If RaaS runs on a remote node
# (2b_raas_b200.sh), export ASTRAFLOW_RAAS_URL=http://<b200-host>:19190.
#
# Usage (terminal 3, after AstraFlow and RaaS are ready):
#   bash examples/math/qwen3-30b-a3b-m2po/scripts/3_trainer_model0.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
# Export EXP_NAME and TRIAL_NAME from the experiment YAML.
astraflow_load_experiment_env

# All 6 GPUs: world = tp(2) * pp(1) * dp(3) = 6, ep=2 nested in dp.
export CUDA_VISIBLE_DEVICES="${TRAINER_MODEL0_GPUS:-0,1,2,3,4,5}"
TRAINER0_NPROC="$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')"

export RAAS_PORT="${RAAS_PORT:-19190}"
export ASTRAFLOW_PORT="${ASTRAFLOW_PORT:-8000}"
export ASTRAFLOW_URL="${ASTRAFLOW_URL:-http://127.0.0.1:${ASTRAFLOW_PORT}}"
export ASTRAFLOW_RAAS_URL="${ASTRAFLOW_RAAS_URL:-http://127.0.0.1:${RAAS_PORT}}"

# sender_agent (in trainer) listens on this HTTP port
export WEIGHT_TRANSFER_HTTP_PORT="${WEIGHT_TRANSFER_HTTP_PORT_MODEL0:-19861}"

# NCCL / PYTORCH / WANDB tweaks + LOG_DIR. Defined in examples/_common/utils.sh.
astraflow_setup_env

echo "=== Trainer model0 (TCP, Megatron MoE) ==="
echo "Experiment config   : ${EXPERIMENT_CONFIG}"
echo "GPUs                : ${CUDA_VISIBLE_DEVICES} (Megatron TP2/EP2/DP$((TRAINER0_NPROC / 2)))"
echo "AstraFlow           : ${ASTRAFLOW_URL}"
echo "RaaS                : ${ASTRAFLOW_RAAS_URL}"
echo "Sender HTTP         : ${WEIGHT_TRANSFER_HTTP_PORT}"
echo "WANDB mode          : ${WANDB_MODE:-online}"
echo "=========================================="

torchrun --nnodes 1 --nproc-per-node "${TRAINER0_NPROC}" \
  --master-addr "${MASTER_ADDR:-127.0.0.1}" --master-port "${MASTER_PORT_MODEL0:-29541}" \
  examples/launch_trainer.py \
  --config "${EXPERIMENT_CONFIG}" \
  --trainer trainer_model0 \
  "$@" 2>&1 | tee "${LOG_DIR}/trainer_model0.log"
