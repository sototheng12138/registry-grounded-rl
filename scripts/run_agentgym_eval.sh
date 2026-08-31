#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
AGENTGYM_ROOT="${AGENTGYM_ROOT:?Set AGENTGYM_ROOT to the pinned AgentGym-RL checkout}"
EVAL_PYTHON="${EVAL_PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
MODEL_TAG="${MODEL_TAG:?MODEL_TAG is required}"
VIEW="${VIEW:?VIEW is required}"
TASK_FILE="${TASK_FILE:-${PROJECT_ROOT}/artifacts/workflow_tasks_v1/dev.jsonl}"
EVAL_DATA="${EVAL_DATA:-${PROJECT_ROOT}/artifacts/workflow_agentgym_eval_dev_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/frozen_dev_eval_v1}"
RUN_SEED="${RUN_SEED:-1701}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
BATCH_SIZE="${BATCH_SIZE:-24}"
MAX_TOKENS="${MAX_TOKENS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MODEL_SHA="${MODEL_SHA:-}"

case "${VIEW}" in
  original|order|schema_surface|opaque_alias|hard_distractor|unavailable) ;;
  *) echo "Unsupported VIEW: ${VIEW}" >&2; exit 2 ;;
esac
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Missing model: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${TASK_FILE}" || ! -f "${EVAL_DATA}/registryworkflow_test.json" ]]; then
  echo "Missing frozen evaluation data" >&2
  exit 2
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver unavailable" >&2
  exit 3
fi

RUN_DIR="${OUTPUT_ROOT}/${MODEL_TAG}/${VIEW}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to mix rerun into ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}/trajectories"

export PYTHONPATH="${PROJECT_ROOT}:${AGENTGYM_ROOT}/AgentGym/agentenv:${AGENTGYM_ROOT}/AgentGym-RL${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=true

if [[ -z "${MODEL_SHA}" ]]; then
  MODEL_SHA="$(${EVAL_PYTHON} - "${MODEL_PATH}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
index = root / "model.safetensors.index.json"
if index.exists():
    files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
else:
    files = ["model.safetensors"]
digest = hashlib.sha256()
for name in files:
    path = root / name
    digest.update(name.encode())
    digest.update(str(path.stat().st_size).encode())
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
print(digest.hexdigest())
PY
)"
fi

${EVAL_PYTHON} - "${RUN_DIR}" "${MODEL_PATH}" "${MODEL_TAG}" "${VIEW}" "${TASK_FILE}" "${EVAL_DATA}" "${RUN_SEED}" "${MAX_ROUNDS}" "${BATCH_SIZE}" "${MAX_TOKENS}" "${MODEL_SHA}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir, model, tag, view, task, data, seed, rounds, batch, tokens, model_sha = sys.argv[1:]
task_path = Path(task).resolve(strict=True)
data_path = Path(data).resolve(strict=True)
manifest = {
    "schema_version": "registry-grounded-rl/frozen-dev-eval-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "model": {"path": str(Path(model).resolve(strict=True)), "tag": tag, "sha256": model_sha},
    "view": view,
    "task_file": {"path": str(task_path), "sha256": hashlib.sha256(task_path.read_bytes()).hexdigest()},
    "eval_data": {"path": str(data_path), "manifest": json.loads((data_path / "registryworkflow_test.manifest.json").read_text())},
    "seed": int(seed),
    "decoding": {"temperature": 0.0, "top_p": 1.0, "max_tokens_per_turn": int(tokens)},
    "max_rounds": int(rounds),
    "batch_size": int(batch),
}
Path(run_dir, "eval_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

cd "${AGENTGYM_ROOT}/AgentGym-RL"
HYDRA_FULL_ERROR=1 "${EVAL_PYTHON}" -m verl.agent_trainer.main_generation \
  data.path="${EVAL_DATA}" \
  data.max_prompt_length=2048 \
  data.max_response_length=4096 \
  data.n_samples=1 \
  data.batch_size="${BATCH_SIZE}" \
  agentgym.task_name=registryworkflow \
  agentgym.env_addr="'workflow://${TASK_FILE}?view=${VIEW}&seed=${RUN_SEED}'" \
  agentgym.max_rounds="${MAX_ROUNDS}" \
  agentgym.timeout=600 \
  model.path="${MODEL_PATH}" \
  rollout.temperature=0.0 \
  rollout.top_p=1.0 \
  rollout.max_tokens="${MAX_TOKENS}" \
  rollout.max_model_len=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  rollout.rollout_log_dir="${RUN_DIR}/trajectories" \
  rollout.send_interval=0 \
  trainer.n_gpus_per_node=1 \
  2>&1 | tee "${RUN_DIR}/eval.log"

"${EVAL_PYTHON}" - "${RUN_DIR}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve(strict=True)
files = sorted((run_dir / "trajectories").rglob("*.json"))
episodes = 0
reward = 0.0
trajectory_hashes = {}
for path in files:
    rows = json.loads(path.read_text())
    episodes += len(rows)
    reward += sum(float(row["reward"]) for row in rows)
    trajectory_hashes[str(path.relative_to(run_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
completion = {
    "episodes": episodes,
    "reward_sum": reward,
    "success_rate": reward / episodes if episodes else None,
    "eval_log_sha256": hashlib.sha256((run_dir / "eval.log").read_bytes()).hexdigest(),
    "trajectory_sha256": trajectory_hashes,
}
(run_dir / "completed.json").write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
print(json.dumps(completion, indent=2, sort_keys=True))
PY
