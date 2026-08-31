"""CPU-only commands for data construction and environment validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from .dataset import build_splits, load_tasks
from .environment import RegistryGroundedEnv, RegistryView
from .evaluation import evaluate_episodes
from .oracle import run_name_memorizer, run_oracle
from .tasks import generate_tasks
from .workflow_environment import RegistryWorkflowEnv
from .workflow_oracle import run_workflow_name_memorizer, run_workflow_oracle
from .workflow_tasks import generate_workflow_tasks


def _views() -> tuple[RegistryView, ...]:
    return tuple(RegistryView)


def _run_smoke(count: int, seed: int) -> dict[str, object]:
    tasks = generate_tasks(count, seed=seed, split="smoke", template_families=(0, 1, 2, 3))
    oracle_rows = [
        run_oracle(RegistryGroundedEnv(task, view)) for task in tasks for view in _views()
    ]
    memorizer_rows = [
        run_name_memorizer(RegistryGroundedEnv(task, view))
        for task in tasks
        for view in _views()
    ]
    return {
        "tasks": count,
        "seed": seed,
        "oracle": evaluate_episodes(oracle_rows),
        "name_memorizer": evaluate_episodes(memorizer_rows),
    }


def _run_workflow_smoke(count: int, seed: int) -> dict[str, object]:
    tasks = generate_workflow_tasks(
        count,
        seed=seed,
        split="smoke",
        template_families=(0, 1, 2, 3),
    )
    oracle_rows = [
        run_workflow_oracle(RegistryWorkflowEnv(task, view))
        for task in tasks
        for view in _views()
    ]
    memorizer_rows = [
        run_workflow_name_memorizer(RegistryWorkflowEnv(task, view))
        for task in tasks
        for view in _views()
    ]
    return {
        "tasks": count,
        "seed": seed,
        "oracle": evaluate_episodes(oracle_rows),
        "name_memorizer": evaluate_episodes(memorizer_rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="Validate all views with oracle and brittle policy")
    smoke.add_argument("--count", type=int, default=12)
    smoke.add_argument("--seed", type=int, default=1701)
    smoke.add_argument("--output")

    workflow_smoke = subparsers.add_parser(
        "workflow-smoke", help="Validate stateful workflow views and exact end-state scoring"
    )
    workflow_smoke.add_argument("--count", type=int, default=12)
    workflow_smoke.add_argument("--seed", type=int, default=1701)
    workflow_smoke.add_argument("--output")

    build = subparsers.add_parser("build", help="Freeze deterministic train/dev/test task files")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--train-count", type=int, default=1024)
    build.add_argument("--dev-count", type=int, default=128)
    build.add_argument("--test-count", type=int, default=256)
    build.add_argument("--seed", type=int, default=1701)

    workflow_build = subparsers.add_parser(
        "workflow-build", help="Freeze deterministic stateful workflow task files"
    )
    workflow_build.add_argument("--output-dir", required=True)
    workflow_build.add_argument("--train-count", type=int, default=512)
    workflow_build.add_argument("--dev-count", type=int, default=64)
    workflow_build.add_argument("--test-count", type=int, default=128)
    workflow_build.add_argument("--seed", type=int, default=1701)

    workflow_prefix = subparsers.add_parser(
        "workflow-prefix", help="Freeze a model-independent prefix for optimizer smoke"
    )
    workflow_prefix.add_argument("--source", required=True)
    workflow_prefix.add_argument("--output-dir", required=True)
    workflow_prefix.add_argument("--count", type=int, default=16)

    validate = subparsers.add_parser("validate", help="Load and validate a task JSONL")
    validate.add_argument("--tasks", required=True)

    workflow_validate = subparsers.add_parser(
        "workflow-validate", help="Load and validate a workflow JSONL"
    )
    workflow_validate.add_argument("--tasks", required=True)

    workflow_controls = subparsers.add_parser(
        "workflow-controls", help="Evaluate oracle and canonical-name controls on frozen tasks"
    )
    workflow_controls.add_argument("--tasks", required=True)
    workflow_controls.add_argument("--count", type=int)
    workflow_controls.add_argument("--output", required=True)

    agentgym_index = subparsers.add_parser(
        "agentgym-index", help="Create AgentGym-RL item IDs for a frozen task split"
    )
    agentgym_index.add_argument("--tasks", required=True)
    agentgym_index.add_argument("--output", required=True)

    workflow_index = subparsers.add_parser(
        "workflow-agentgym-index", help="Create AgentGym-RL IDs for a workflow split"
    )
    workflow_index.add_argument("--tasks", required=True)
    workflow_index.add_argument("--output", required=True)

    numeric_splits = subparsers.add_parser(
        "agentgym-numeric-splits", help="Freeze item-ID splits for a stateful AgentGym task"
    )
    numeric_splits.add_argument("--output-dir", required=True)
    numeric_splits.add_argument("--prefix", required=True)
    numeric_splits.add_argument("--count", type=int, required=True)
    numeric_splits.add_argument("--train-count", type=int, required=True)
    numeric_splits.add_argument("--dev-count", type=int, required=True)
    numeric_splits.add_argument("--seed", type=int, default=1701)

    model_smoke = subparsers.add_parser("model-smoke", help="Run one local Qwen episode")
    model_smoke.add_argument("--model-path", required=True)
    model_smoke.add_argument("--view", choices=[view.value for view in RegistryView], default="original")
    model_smoke.add_argument("--seed", type=int, default=1701)
    model_smoke.add_argument("--device", default="auto")
    model_smoke.add_argument("--dtype", default="auto")
    model_smoke.add_argument("--max-new-tokens", type=int, default=160)
    model_smoke.add_argument("--output")

    model_batch = subparsers.add_parser(
        "model-batch", help="Run a small frozen local-Qwen batch across registry views"
    )
    model_batch.add_argument("--model-path", required=True)
    model_batch.add_argument("--tasks", required=True)
    model_batch.add_argument("--count", type=int, default=4)
    model_batch.add_argument(
        "--views",
        nargs="+",
        choices=[view.value for view in RegistryView],
        default=[view.value for view in RegistryView],
    )
    model_batch.add_argument("--device", default="auto")
    model_batch.add_argument("--dtype", default="auto")
    model_batch.add_argument("--max-new-tokens", type=int, default=160)
    model_batch.add_argument("--output", required=True)

    workflow_model_batch = subparsers.add_parser(
        "workflow-model-batch", help="Run a frozen local-Qwen workflow batch across views"
    )
    workflow_model_batch.add_argument("--model-path", required=True)
    workflow_model_batch.add_argument("--tasks", required=True)
    workflow_model_batch.add_argument("--count", type=int, default=4)
    workflow_model_batch.add_argument(
        "--views",
        nargs="+",
        choices=[view.value for view in RegistryView],
        default=[view.value for view in RegistryView],
    )
    workflow_model_batch.add_argument("--device", default="auto")
    workflow_model_batch.add_argument("--dtype", default="auto")
    workflow_model_batch.add_argument("--max-new-tokens", type=int, default=192)
    workflow_model_batch.add_argument("--output", required=True)

    rescore = subparsers.add_parser(
        "rescore-model-artifact", help="Recompute summary metrics without rerunning inference"
    )
    rescore.add_argument("--input", required=True)
    rescore.add_argument("--output", required=True)

    combine = subparsers.add_parser(
        "combine-model-artifacts", help="Combine disjoint trajectory slices and rescore"
    )
    combine.add_argument("--inputs", nargs="+", required=True)
    combine.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        result = _run_smoke(args.count, args.seed)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    if args.command == "workflow-smoke":
        result = _run_workflow_smoke(args.count, args.seed)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    if args.command == "build":
        result = build_splits(
            args.output_dir,
            train_count=args.train_count,
            dev_count=args.dev_count,
            test_count=args.test_count,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "workflow-build":
        from .workflow_dataset import build_workflow_splits

        result = build_workflow_splits(
            args.output_dir,
            train_count=args.train_count,
            dev_count=args.dev_count,
            test_count=args.test_count,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "workflow-prefix":
        from .workflow_dataset import freeze_workflow_prefix

        result = freeze_workflow_prefix(args.source, args.output_dir, count=args.count)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        tasks = load_tasks(args.tasks)
        print(json.dumps({"path": args.tasks, "rows": len(tasks), "valid": True}, indent=2))
        return 0
    if args.command == "workflow-validate":
        from .workflow_dataset import load_workflow_tasks

        tasks = load_workflow_tasks(args.tasks)
        print(json.dumps({"path": args.tasks, "rows": len(tasks), "valid": True}, indent=2))
        return 0
    if args.command == "workflow-controls":
        from .workflow_dataset import load_workflow_tasks

        source = Path(args.tasks).expanduser().resolve(strict=True)
        tasks = load_workflow_tasks(source)
        if args.count is not None:
            if args.count <= 0:
                raise ValueError("count must be positive")
            tasks = tasks[: args.count]
        oracle_rows = [
            run_workflow_oracle(RegistryWorkflowEnv(task, view))
            for task in tasks
            for view in _views()
        ]
        memorizer_rows = [
            run_workflow_name_memorizer(RegistryWorkflowEnv(task, view))
            for task in tasks
            for view in _views()
        ]
        result = {
            "task_source": str(source),
            "task_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "selected_tasks": len(tasks),
            "oracle": evaluate_episodes(oracle_rows),
            "name_memorizer": evaluate_episodes(memorizer_rows),
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "agentgym-index":
        from .agentgym_data import build_agentgym_index

        result = build_agentgym_index(args.tasks, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "workflow-agentgym-index":
        from .agentgym_data import build_workflow_agentgym_index

        result = build_workflow_agentgym_index(args.tasks, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "agentgym-numeric-splits":
        from .agentgym_data import build_numeric_item_splits

        result = build_numeric_item_splits(
            args.output_dir,
            prefix=args.prefix,
            count=args.count,
            train_count=args.train_count,
            dev_count=args.dev_count,
            seed=args.seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "model-smoke":
        from .local_qwen import LocalQwenAgent

        task = generate_tasks(1, seed=args.seed, split="smoke", template_families=(0,))[0]
        agent = LocalQwenAgent(
            args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
        trajectory = agent.run(RegistryGroundedEnv(task, RegistryView(args.view)))
        rendered = json.dumps(trajectory, ensure_ascii=False, indent=2, sort_keys=True)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0
    if args.command == "model-batch":
        from .local_qwen import LocalQwenAgent

        tasks = load_tasks(args.tasks)[: args.count]
        if not tasks:
            raise ValueError("No tasks selected")
        agent = LocalQwenAgent(
            args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
        trajectories = [
            agent.run(RegistryGroundedEnv(task, RegistryView(view)))
            for task in tasks
            for view in args.views
        ]
        result = {
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "task_source": str(Path(args.tasks).expanduser().resolve()),
            "selected_tasks": len(tasks),
            "views": list(args.views),
            "summary": evaluate_episodes(trajectories),
            "trajectories": trajectories,
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "workflow-model-batch":
        from .local_qwen import LocalQwenAgent
        from .workflow_dataset import load_workflow_tasks

        tasks = load_workflow_tasks(args.tasks)[: args.count]
        if not tasks:
            raise ValueError("No workflow tasks selected")
        agent = LocalQwenAgent(
            args.model_path,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
        )
        trajectories = [
            agent.run_workflow(RegistryWorkflowEnv(task, RegistryView(view)))
            for task in tasks
            for view in args.views
        ]
        result = {
            "model_path": str(Path(args.model_path).expanduser().resolve()),
            "task_source": str(Path(args.tasks).expanduser().resolve()),
            "selected_tasks": len(tasks),
            "views": list(args.views),
            "summary": evaluate_episodes(trajectories),
            "trajectories": trajectories,
        }
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "rescore-model-artifact":
        source = Path(args.input).expanduser().resolve(strict=True)
        destination = Path(args.output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(destination)
        value = json.loads(source.read_text(encoding="utf-8"))
        trajectories = value.get("trajectories")
        if not isinstance(trajectories, list) or not trajectories:
            raise ValueError("Input artifact has no non-empty trajectories list")
        value["summary"] = evaluate_episodes(trajectories)
        value["rescored_from"] = str(source)
        value["rescored_from_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        value["summary_schema"] = "registry-grounded-rl/evaluation-v2"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(value["summary"], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "combine-model-artifacts":
        destination = Path(args.output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(destination)
        sources: list[dict[str, str]] = []
        trajectories: list[dict[str, object]] = []
        model_paths: set[str] = set()
        task_sources: set[str] = set()
        for input_path in args.inputs:
            source = Path(input_path).expanduser().resolve(strict=True)
            value = json.loads(source.read_text(encoding="utf-8"))
            rows = value.get("trajectories")
            if not isinstance(rows, list) or not rows:
                raise ValueError(f"No trajectories in {source}")
            trajectories.extend(rows)
            sources.append(
                {
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            if isinstance(value.get("model_path"), str):
                model_paths.add(value["model_path"])
            if isinstance(value.get("task_source"), str):
                task_sources.add(value["task_source"])
        if len(model_paths) > 1 or len(task_sources) > 1:
            raise ValueError("Combined slices must share model_path and task_source")
        result = {
            "schema_version": "registry-grounded-rl/combined-model-artifact-v1",
            "sources": sources,
            "model_path": next(iter(model_paths), None),
            "task_source": next(iter(task_sources), None),
            "summary": evaluate_episodes(trajectories),
            "trajectories": trajectories,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
