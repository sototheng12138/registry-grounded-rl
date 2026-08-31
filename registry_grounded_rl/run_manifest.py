"""Write-once provenance manifest for a RegistryGrounded-RL training run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_weights_digest(model_path: Path) -> tuple[str | None, list[str]]:
    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        names = sorted(set(weight_map.values()))
    elif (model_path / "model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        return None, []
    digest = hashlib.sha256()
    for name in names:
        path = model_path / name
        digest.update(name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), names


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _code_digest(project_root: Path) -> tuple[str, list[str]]:
    paths = sorted(
        [
            *(project_root / "registry_grounded_rl").glob("*.py"),
            *project_root.glob("scripts/*.sh"),
            *project_root.glob("scripts/*.py"),
        ]
    )
    digest = hashlib.sha256()
    relative_paths: list[str] = []
    for path in paths:
        relative = str(path.relative_to(project_root))
        relative_paths.append(relative)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def _hardware() -> dict[str, Any]:
    value: dict[str, Any] = {}
    try:
        import torch
    except ImportError:
        return {"torch_importable": False}
    value.update(
        {
            "torch_importable": True,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        }
    )
    if torch.cuda.is_available():
        value["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return value


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.expanduser().resolve(strict=True)
    agentgym_root = args.agentgym_root.expanduser().resolve(strict=True)
    task_file = args.task_file.expanduser().resolve(strict=True)
    index_file = args.index_file.expanduser().resolve(strict=True)
    model_path = args.model_path.expanduser().resolve(strict=True)
    code_sha256, code_files = _code_digest(project_root)
    model_config = model_path / "config.json"
    model_weights_sha256, model_weight_files = _model_weights_digest(model_path)
    agentgym_client_patch = (
        agentgym_root / "AgentGym-RL" / "verl" / "utils" / "agentgym" / "client.py"
    )
    agentgym_rollout_patch = (
        agentgym_root
        / "AgentGym-RL"
        / "verl"
        / "workers"
        / "rollout"
        / "agent_vllm_rollout"
        / "vllm_rollout.py"
    )
    return {
        "schema_version": "registry-grounded-rl/run-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": args.arm,
        "benchmark": args.benchmark,
        "run_seed": args.run_seed,
        "environment_address": args.environment_address,
        "training": {
            "actor_lr": args.actor_lr,
            "data_seed": args.run_seed,
            "entropy_coeff": args.entropy_coeff,
            "kl_loss_coef": args.kl_loss_coef,
            "rollout_seed": args.run_seed,
            "n_gpus": args.n_gpus,
            "train_batch_size": args.train_batch_size,
            "rollout_n": args.rollout_n,
            "total_epochs": args.total_epochs,
            "total_training_steps": args.total_training_steps,
            "max_rounds": args.max_rounds,
            "algorithm": "grpo",
            "terminal_reward": "binary-environment-reward",
            "sampling_seed_strategy": "independent-per-trajectory-round-v1",
        },
        "task_file": {"path": str(task_file), "sha256": _sha256(task_file)},
        "index_file": {"path": str(index_file), "sha256": _sha256(index_file)},
        "model": {
            "path": str(model_path),
            "config_sha256": _sha256(model_config) if model_config.is_file() else None,
            "weights_sha256": model_weights_sha256,
            "weight_files": model_weight_files,
        },
        "source": {
            "project_root": str(project_root),
            "code_sha256": code_sha256,
            "code_files": code_files,
            "agentgym_rl_commit": _git_commit(agentgym_root),
            "agentgym_submodule_commit": _git_commit(agentgym_root / "AgentGym"),
            "agentgym_client_sha256": (
                _sha256(agentgym_client_patch) if agentgym_client_patch.is_file() else None
            ),
            "agentgym_rollout_sha256": (
                _sha256(agentgym_rollout_patch) if agentgym_rollout_patch.is_file() else None
            ),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            **_hardware(),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--agentgym-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--environment-address", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--run-seed", type=int, required=True)
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--total-training-steps", type=int)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--entropy-coeff", type=float, default=0.001)
    parser.add_argument("--kl-loss-coef", type=float, default=0.001)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
