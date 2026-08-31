"""Frozen split writer with content hashes and no model-dependent selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .tasks import TaskSpec, generate_tasks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, tasks: Iterable[TaskSpec]) -> int:
    rows = list(tasks)
    with path.open("x", encoding="utf-8") as handle:
        for task in rows:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def build_splits(
    output_dir: str | Path,
    *,
    train_count: int,
    dev_count: int,
    test_count: int,
    seed: int,
) -> dict[str, Any]:
    """Write train/dev/test once; refuse to overwrite any existing directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    specs = {
        "train": (train_count, seed, (0, 1)),
        "dev": (dev_count, seed + 1, (2,)),
        "test": (test_count, seed + 2, (3,)),
    }
    files: dict[str, Any] = {}
    all_ids: set[str] = set()
    for split, (count, split_seed, families) in specs.items():
        tasks = generate_tasks(
            count,
            seed=split_seed,
            split=split,
            template_families=families,
        )
        ids = {task.task_id for task in tasks}
        if all_ids & ids:
            raise RuntimeError("Task IDs overlap across splits")
        all_ids.update(ids)
        path = destination / f"{split}.jsonl"
        written = _write_jsonl(path, tasks)
        files[split] = {"path": path.name, "rows": written, "sha256": _sha256(path)}
    manifest = {
        "schema_version": "registry-grounded-rl/tasks-v1",
        "seed": seed,
        "generator": "deterministic-python-random",
        "model_outputs_used": False,
        "files": files,
    }
    manifest_path = destination / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest | {"manifest_sha256": _sha256(manifest_path)}


def load_tasks(path: str | Path) -> tuple[TaskSpec, ...]:
    rows: list[TaskSpec] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            try:
                rows.append(TaskSpec.from_dict(value))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid task at line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("Task file is empty")
    return tuple(rows)

