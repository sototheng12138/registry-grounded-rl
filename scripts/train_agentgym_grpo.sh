#!/usr/bin/env bash
set -euo pipefail

# Matched B3/B4 launcher for the pinned AgentGym-RL fork.
# Required before launch: working NVIDIA driver and a local Qwen2.5 checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
AGENTGYM_ROOT="${AGENTGYM_ROOT:?Set AGENTGYM_ROOT to the pinned AgentGym-RL checkout}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to Qwen2.5-3B-Instruct or a stage-2 checkpoint}"
BENCHMARK="${BENCHMARK:-workflow}"
TASK_FILE="${TASK_FILE:-}"
INDEX_FILE="${INDEX_FILE:-}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:36001}"
ARM="${ARM:-original}"
RUN_SEED="${RUN_SEED:-1701}"
N_GPUS="${N_GPUS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
SAVE_FREQ="${SAVE_FREQ:-10}"
RUN_TAG="${RUN_TAG:-}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"

MANIFEST_STEP_ARGS=()
TRAINER_STEP_ARGS=()
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  MANIFEST_STEP_ARGS=(--total-training-steps "${TOTAL_TRAINING_STEPS}")
  TRAINER_STEP_ARGS=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

if [[ "${ARM}" != "original" && "${ARM}" != "orbit" && "${ARM}" != "capability_orbit" && "${ARM}" != "grouped_capability_orbit" && "${ARM}" != "stratified_capability_orbit" && "${ARM}" != "selective_capability_orbit" && "${ARM}" != "stratified_solvable_orbit" ]]; then
  echo "Unsupported ARM: ${ARM}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing model checkpoint: ${MODEL_PATH}" >&2
  exit 2
fi
TASK_NAME="registrygrounded"
if [[ "${BENCHMARK}" == "contract" ]]; then
  TASK_FILE="${TASK_FILE:-${PROJECT_ROOT}/artifacts/tasks_dev_v0/train.jsonl}"
  ENV_ADDR="inproc://${TASK_FILE}?view=${ARM}&seed=${RUN_SEED}"
  INDEX_FILE="${INDEX_FILE:-${PROJECT_ROOT}/artifacts/agentgym_train_v0.jsonl}"
  if [[ ! -f "${TASK_FILE}" ]]; then
    echo "Missing frozen task file: ${TASK_FILE}" >&2
    exit 2
  fi
elif [[ "${BENCHMARK}" == "workflow" ]]; then
  TASK_NAME="registryworkflow"
  TASK_FILE="${TASK_FILE:-${PROJECT_ROOT}/artifacts/workflow_tasks_v1/train.jsonl}"
  INDEX_FILE="${INDEX_FILE:-${PROJECT_ROOT}/artifacts/workflow_agentgym_train_v1.jsonl}"
  ENV_ADDR="workflow://${TASK_FILE}?view=${ARM}&seed=${RUN_SEED}&group_size=${ROLLOUT_N}"
  if [[ ! -f "${TASK_FILE}" ]]; then
    echo "Missing frozen workflow task file: ${TASK_FILE}" >&2
    exit 2
  fi
elif [[ "${BENCHMARK}" == "todo" || "${BENCHMARK}" == "weather" ]]; then
  TASK_NAME="registry${BENCHMARK}"
  INDEX_FILE="${INDEX_FILE:-${PROJECT_ROOT}/artifacts/agentgym_${BENCHMARK}_v0/train.jsonl}"
  ENV_ADDR="registry+${SERVER_URL}?view=${ARM}&seed=${RUN_SEED}"
  if [[ "${BENCHMARK}" == "todo" && "${ALLOW_DESTRUCTIVE_TODO:-0}" != "1" ]]; then
    echo "Todo reset deletes and rebuilds account projects; use only a disposable account and set ALLOW_DESTRUCTIVE_TODO=1." >&2
    exit 2
  fi
  if [[ "${BENCHMARK}" == "weather" && "${ALLOW_LIVE_WEATHER:-0}" != "1" ]]; then
    echo "Weather uses a drifting live API; it is external validation only. Set ALLOW_LIVE_WEATHER=1 to override." >&2
    exit 2
  fi
else
  echo "BENCHMARK must be workflow, contract, todo, or weather" >&2
  exit 2
fi
if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "Missing AgentGym index: ${INDEX_FILE}" >&2
  exit 2
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver is unavailable; refusing to label a CPU-only run as GRPO training." >&2
  exit 3
fi

export PYTHONPATH="${PROJECT_ROOT}:${AGENTGYM_ROOT}/AgentGym/agentenv:${AGENTGYM_ROOT}/AgentGym-RL${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE="${WANDB_MODE:-offline}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/agentgym_grpo}"
RUN_NAME="${BENCHMARK}_b3_original_seed${RUN_SEED}"
if [[ "${ARM}" == "orbit" ]]; then
  RUN_NAME="${BENCHMARK}_b4_orbit_seed${RUN_SEED}"
elif [[ "${ARM}" == "capability_orbit" ]]; then
  RUN_NAME="${BENCHMARK}_b5_capability_orbit_seed${RUN_SEED}"
elif [[ "${ARM}" == "grouped_capability_orbit" ]]; then
  RUN_NAME="${BENCHMARK}_b6_grouped_capability_orbit_seed${RUN_SEED}"
elif [[ "${ARM}" == "stratified_capability_orbit" ]]; then
  RUN_NAME="${BENCHMARK}_b7_stratified_capability_orbit_seed${RUN_SEED}"
elif [[ "${ARM}" == "selective_capability_orbit" ]]; then
  RUN_NAME="${BENCHMARK}_b8_selective_capability_orbit_seed${RUN_SEED}"
elif [[ "${ARM}" == "stratified_solvable_orbit" ]]; then
  RUN_NAME="${BENCHMARK}_b9_stratified_solvable_orbit_seed${RUN_SEED}"
fi
if [[ -n "${RUN_TAG}" ]]; then
  RUN_NAME="${RUN_NAME}_${RUN_TAG}"
fi
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Run directory already exists; refusing to mix a rerun into ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

"${TRAIN_PYTHON}" -m registry_grounded_rl.run_manifest \
  --output "${RUN_DIR}/run_manifest.json" \
  --project-root "${PROJECT_ROOT}" \
  --agentgym-root "${AGENTGYM_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --task-file "${TASK_FILE}" \
  --index-file "${INDEX_FILE}" \
  --environment-address "${ENV_ADDR}" \
  --arm "${ARM}" \
  --benchmark "${BENCHMARK}" \
  --run-seed "${RUN_SEED}" \
  --n-gpus "${N_GPUS}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --rollout-n "${ROLLOUT_N}" \
  --total-epochs "${TOTAL_EPOCHS}" \
  --max-rounds "${MAX_ROUNDS}" \
  --actor-lr "${ACTOR_LR}" \
  --entropy-coeff "${ENTROPY_COEFF}" \
  --kl-loss-coef "${KL_LOSS_COEF}" \
  "${MANIFEST_STEP_ARGS[@]}"
"${TRAIN_PYTHON}" -m pip freeze > "${RUN_DIR}/environment_freeze.txt"

cd "${AGENTGYM_ROOT}/AgentGym-RL"
HYDRA_FULL_ERROR=1 "${TRAIN_PYTHON}" -m verl.agent_trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.rounds_ctrl.type=fixed \
  algorithm.rounds_ctrl.rounds="${MAX_ROUNDS}" \
  data.train_file="${INDEX_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  +data.seed="${RUN_SEED}" \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.return_raw_chat=True \
  actor_rollout_ref.agentgym.task_name="${TASK_NAME}" \
  actor_rollout_ref.agentgym.env_addr="'${ENV_ADDR}'" \
  actor_rollout_ref.agentgym.timeout=600 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.top_p=0.95 \
  +actor_rollout_ref.rollout.seed="${RUN_SEED}" \
  actor_rollout_ref.rollout.max_tokens=256 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.rollout_log_dir="${RUN_DIR}/trajectories" \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.project_name=registry-grounded-rl \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${RUN_DIR}/checkpoints" \
  trainer.logger='[console]' \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "${TRAINER_STEP_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/train.log"
