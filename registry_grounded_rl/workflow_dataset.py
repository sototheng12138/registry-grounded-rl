"""Frozen writer and loader for stateful workflow tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .workflow_tasks import WorkflowTaskSpec, generate_workflow_tasks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, tasks: Iterable[WorkflowTaskSpec]) -> int:
    rows = tuple(tasks)
    with path.open("x", encoding="utf-8") as handle:
        for task in rows:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def build_workflow_splits(
    output_dir: str | Path,
    *,
    train_count: int,
    dev_count: int,
    test_count: int,
    seed: int,
) -> dict[str, Any]:
    """Freeze disjoint workflow splits and refuse accidental overwrite."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    specs = {
        "train": (train_count, seed, (0, 1)),
        "dev": (dev_count, seed + 1, (2,)),
        "test": (test_count, seed + 2, (3,)),
    }
    files: dict[str, dict[str, Any]] = {}
    all_ids: set[str] = set()
    for split, (count, split_seed, families) in specs.items():
        tasks = generate_workflow_tasks(
            count,
            seed=split_seed,
            split=split,
            template_families=families,
        )
        ids = {task.task_id for task in tasks}
        if all_ids & ids:
            raise RuntimeError("Task IDs overlap across workflow splits")
        all_ids.update(ids)
        path = destination / f"{split}.jsonl"
        written = _write_jsonl(path, tasks)
        files[split] = {"path": path.name, "rows": written, "sha256": _sha256(path)}
    manifest = {
        "schema_version": "registry-grounded-rl/stateful-workflow-v1",
        "seed": seed,
        "generator": "deterministic-python-random",
        "model_outputs_used": False,
        "live_services_used": False,
        "scoring": "exact terminal workspace state; unrelated records must remain unchanged",
        "files": files,
    }
    manifest_path = destination / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest | {"manifest_sha256": _sha256(manifest_path)}


def load_workflow_tasks(path: str | Path) -> tuple[WorkflowTaskSpec, ...]:
    rows: list[WorkflowTaskSpec] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(WorkflowTaskSpec.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid workflow task at line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("Workflow task file is empty")
    return tuple(rows)


def freeze_workflow_prefix(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    count: int,
) -> dict[str, Any]:
    """Freeze a model-independent train prefix for pipeline smoke runs."""

    if count <= 0:
        raise ValueError("count must be positive")
    source = Path(source_path).expanduser().resolve(strict=True)
    tasks = load_workflow_tasks(source)
    if count >= len(tasks):
        raise ValueError("prefix count must be smaller than the source split")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    task_path = destination / "train.jsonl"
    rows = _write_jsonl(task_path, tasks[:count])
    manifest = {
        "schema_version": "registry-grounded-rl/stateful-workflow-prefix-v1",
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "selection": "first-N-before-model-evaluation",
        "model_outputs_used": False,
        "rows": rows,
        "file": {"path": task_path.name, "sha256": _sha256(task_path)},
    }
    manifest_path = destination / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest | {"manifest_sha256": _sha256(manifest_path)}
