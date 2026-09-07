#!/bin/bash
set -euo pipefail
# [2b/3] Launch RaaS on the remote 4xB200 rollout node (SGLang dp=4 tp=1)
#
# Run this ON THE B200 NODE, after 1_astraflow.sh is up on the trainer
# node. ASTRAFLOW_URL must point at the trainer node and is required
# (there is no sane loopback default for a remote node).
#
# Cross-node reachability (see yaml/raas_b200.yaml header):
#   B200 -> trainer : 8000 (AstraFlow), 19861 (sender HTTP), 21000 (handshake)
#   trainer -> B200 : 19190 (RaaS)
# Hostnames must resolve both ways; otherwise export IPs explicitly.
#
# Usage (on the B200 node):
#   ASTRAFLOW_URL=http://<trainer-host>:8000 \
#     bash examples/math/qwen3-30b-a3b-m2po/scripts/2b_raas_b200.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YAML_DIR="${SCRIPT_DIR}/yaml"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${YAML_DIR}/experiment.yaml}"
export RAAS_CONFIG="${RAAS_CONFIG:-${YAML_DIR}/raas_b200.yaml}"
source "${REPO_ROOT}/examples/_common/utils.sh"
# Export EXP_NAME and TRIAL_NAME from the experiment YAML.
astraflow_load_experiment_env

if [ -z "${ASTRAFLOW_URL:-}" ]; then
  echo "ERROR: ASTRAFLOW_URL is not set. On a remote rollout node it must" >&2
  echo "point at the trainer node, e.g.:" >&2
  echo "  ASTRAFLOW_URL=http://<trainer-host>:8000 bash $0" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${SERVICE_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RAAS_HOST="${RAAS_HOST:-0.0.0.0}"
export RAAS_PORT="${RAAS_PORT:-19190}"

# NCCL / PYTORCH / WANDB tweaks + LOG_DIR. Defined in examples/_common/utils.sh.
astraflow_setup_env

echo "=== RaaS Inference Server (B200 remote rollout, R3) ==="
echo "Experiment config : ${EXPERIMENT_CONFIG}"
echo "RaaS config       : ${RAAS_CONFIG}"
echo "GPUs              : ${CUDA_VISIBLE_DEVICES}"
echo "Port              : ${RAAS_PORT}"
echo "AstraFlow URL     : ${ASTRAFLOW_URL}"
echo "======================================================="

python3 -u -m astraflow.raas.server \
  --host "${RAAS_HOST}" \
  --port "${RAAS_PORT}" \
  --config "${EXPERIMENT_CONFIG}" \
  --config "${RAAS_CONFIG}" \
  --engine-id "${ENGINE_ID:-b200-0}" \
  --astraflow-url "${ASTRAFLOW_URL}" \
  2>&1 | tee "${LOG_DIR}/raas_b200.log"
