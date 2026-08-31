#!/usr/bin/env bash
set -euo pipefail

# Evaluate one checkpoint on every frozen workflow registry view.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_TAG="${MODEL_TAG:?MODEL_TAG is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/frozen_eval_matrix}"
BATCH_SIZE="${BATCH_SIZE:-24}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
VIEWS=(original order schema_surface opaque_alias hard_distractor unavailable)

for view in "${VIEWS[@]}"; do
  MODEL_PATH="${MODEL_PATH}" \
  MODEL_TAG="${MODEL_TAG}" \
  VIEW="${view}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    bash "${PROJECT_ROOT}/scripts/run_agentgym_eval.sh"
done
