"""Write the lightweight item-index files consumed by AgentGym-RL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from .dataset import load_tasks
from .workflow_dataset import load_workflow_tasks


def build_agentgym_index(tasks_path: str | Path, output_path: str | Path) -> dict[str, object]:
    tasks = load_tasks(tasks_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for index, task in enumerate(tasks):
            row = {
                "prompt": "registrygrounded",
                "item_id": f"registrygrounded_{index}",
                "extra_info": {"index": index, "task_id": task.task_id},
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"path": str(destination.resolve()), "rows": len(tasks), "sha256": digest}


def build_workflow_agentgym_index(
    tasks_path: str | Path, output_path: str | Path
) -> dict[str, object]:
    tasks = load_workflow_tasks(tasks_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as handle:
        for index, task in enumerate(tasks):
            row = {
                "prompt": "registryworkflow",
                "item_id": f"registryworkflow_{index}",
                "extra_info": {"index": index, "task_id": task.task_id},
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return {"path": str(destination.resolve()), "rows": len(tasks), "sha256": digest}


def build_numeric_item_splits(
    output_dir: str | Path,
    *,
    prefix: str,
    count: int,
    train_count: int,
    dev_count: int,
    seed: int,
) -> dict[str, object]:
    """Freeze model-independent item-id splits for an indexed AgentGym environment."""

    if not prefix or "_" in prefix:
        raise ValueError("prefix must be non-empty and contain no underscore")
    if count <= 0 or min(train_count, dev_count) < 0:
        raise ValueError("counts must be non-negative and total count positive")
    if train_count + dev_count >= count:
        raise ValueError("at least one held-out test item is required")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    item_ids = list(range(count))
    random.Random(seed).shuffle(item_ids)
    split_ids = {
        "train": item_ids[:train_count],
        "dev": item_ids[train_count : train_count + dev_count],
        "test": item_ids[train_count + dev_count :],
    }
    files: dict[str, dict[str, object]] = {}
    for split, ids in split_ids.items():
        path = destination / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as handle:
            for item_id in ids:
                row = {
                    "prompt": prefix,
                    "item_id": f"{prefix}_{item_id}",
                    "extra_info": {"index": item_id, "split": split},
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        files[split] = {
            "path": path.name,
            "rows": len(ids),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": "registry-grounded-rl/agentgym-item-splits-v1",
        "prefix": prefix,
        "source_item_count": count,
        "seed": seed,
        "model_outputs_used": False,
        "files": files,
    }
    manifest_path = destination / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest | {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
