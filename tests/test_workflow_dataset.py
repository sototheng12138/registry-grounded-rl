import json

import pytest

from registry_grounded_rl.agentgym_data import build_workflow_agentgym_index
from registry_grounded_rl.workflow_dataset import (
    build_workflow_splits,
    freeze_workflow_prefix,
    load_workflow_tasks,
)


def test_build_and_load_frozen_workflow_splits(tmp_path) -> None:
    destination = tmp_path / "workflow"
    result = build_workflow_splits(
        destination,
        train_count=8,
        dev_count=3,
        test_count=4,
        seed=1701,
    )
    assert result["files"]["train"]["rows"] == 8
    assert len(load_workflow_tasks(destination / "train.jsonl")) == 8
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["model_outputs_used"] is False
    assert manifest["live_services_used"] is False
    with pytest.raises(FileExistsError):
        build_workflow_splits(
            destination,
            train_count=8,
            dev_count=3,
            test_count=4,
            seed=1701,
        )


def test_workflow_agentgym_index(tmp_path) -> None:
    destination = tmp_path / "workflow"
    build_workflow_splits(
        destination,
        train_count=3,
        dev_count=1,
        test_count=1,
        seed=1701,
    )
    index_path = tmp_path / "train_index.jsonl"
    result = build_workflow_agentgym_index(destination / "train.jsonl", index_path)
    assert result["rows"] == 3
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    assert rows[0]["prompt"] == "registryworkflow"
    assert rows[0]["item_id"] == "registryworkflow_0"


def test_freeze_model_independent_workflow_prefix(tmp_path) -> None:
    destination = tmp_path / "workflow"
    build_workflow_splits(
        destination,
        train_count=8,
        dev_count=2,
        test_count=2,
        seed=1701,
    )
    prefix = tmp_path / "prefix"
    result = freeze_workflow_prefix(destination / "train.jsonl", prefix, count=3)
    assert result["rows"] == 3
    assert result["model_outputs_used"] is False
    assert len(load_workflow_tasks(prefix / "train.jsonl")) == 3
    with pytest.raises(FileExistsError):
        freeze_workflow_prefix(destination / "train.jsonl", prefix, count=3)
