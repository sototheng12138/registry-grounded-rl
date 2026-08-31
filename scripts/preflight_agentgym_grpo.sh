#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
AGENTGYM_ROOT="${AGENTGYM_ROOT:?Set AGENTGYM_ROOT to the pinned AgentGym-RL checkout}"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
CHECK_GPU="${CHECK_GPU:-1}"

status=0

pass() {
  echo "PASS: $1"
}

fail() {
  echo "FAIL: $1" >&2
  status=1
}

check_hash() {
  local path="$1"
  local expected="$2"
  if [[ ! -f "${path}" ]]; then
    fail "missing ${path}"
    return
  fi
  local actual
  actual="$(sha256sum "${path}" | cut -d' ' -f1)"
  if [[ "${actual}" == "${expected}" ]]; then
    pass "hash ${path}"
  else
    fail "hash mismatch ${path}: ${actual}"
  fi
}

if [[ "${CHECK_GPU}" == "1" ]]; then
  if nvidia-smi >/dev/null 2>&1; then
    pass "NVIDIA driver"
  else
    fail "NVIDIA driver unavailable"
  fi
else
  echo "SKIP: NVIDIA driver (CHECK_GPU=${CHECK_GPU})"
fi

check_hash \
  "${PROJECT_ROOT}/artifacts/workflow_tasks_v1/train.jsonl" \
  "d93541e4bd760c34178c3af579022d9ae505ac82ac9155cf767caaf3335e9a4c"
check_hash \
  "${PROJECT_ROOT}/artifacts/workflow_tasks_v1/dev.jsonl" \
  "19719ebac16056fc7bbc756a820bd6cdbd97c15e610881963b3e6bfe14ff3c53"
check_hash \
  "${PROJECT_ROOT}/artifacts/workflow_tasks_v1/test.jsonl" \
  "c12a41ac12a1d4db892c05e1900bad403ba5718915fe1a3e1ff27ccdddd36b19"
check_hash \
  "${PROJECT_ROOT}/artifacts/workflow_agentgym_train_v1.jsonl" \
  "2d703f4756c779ad71042cb60fae386d5e554164aaf9bebe4d8920b1ebdd5f27"

agentgym_commit="$(git -C "${AGENTGYM_ROOT}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${agentgym_commit}" == "82402a99c62a293735a3f412fb8ac9a600673bc0" ]]; then
  pass "AgentGym-RL commit"
else
  fail "AgentGym-RL commit ${agentgym_commit:-missing}"
fi

if rg -q '"registryworkflow": RegistryWorkflowAgentGymClient' \
  "${AGENTGYM_ROOT}/AgentGym-RL/verl/utils/agentgym/client.py"; then
  pass "registryworkflow client registration"
else
  fail "registryworkflow client registration missing"
fi

if env PYTHONPATH="${PROJECT_ROOT}" "${TRAIN_PYTHON}" -c \
  'from registry_grounded_rl.workflow_agentgym import RegistryWorkflowAgentGymClient' \
  >/dev/null 2>&1; then
  pass "project client import"
else
  fail "project client import with ${TRAIN_PYTHON}"
fi

missing_modules="$("${TRAIN_PYTHON}" -c \
  'import importlib.util; names=("hydra","omegaconf","ray","tensordict","torch","vllm"); print(",".join(name for name in names if importlib.util.find_spec(name) is None))' \
  2>/dev/null || echo "python-check-failed")"
if [[ -z "${missing_modules}" ]]; then
  pass "training-stack imports"
else
  fail "training-stack imports with ${TRAIN_PYTHON}; missing ${missing_modules}"
fi

if [[ "${status}" == "0" ]]; then
  echo "READY: AgentGym-RL GRPO preflight passed"
else
  echo "BLOCKED: fix every FAIL item before launching GRPO" >&2
fi
exit "${status}"
