from pathlib import Path

import pytest

from registry_grounded_rl.dataset import build_splits, load_tasks


def test_build_splits_writes_hashed_disjoint_files(tmp_path: Path) -> None:
    output = tmp_path / "data"
    manifest = build_splits(output, train_count=8, dev_count=3, test_count=4, seed=11)
    assert manifest["model_outputs_used"] is False
    assert len(load_tasks(output / "train.jsonl")) == 8
    assert len(load_tasks(output / "dev.jsonl")) == 3
    assert len(load_tasks(output / "test.jsonl")) == 4
    assert all(len(item["sha256"]) == 64 for item in manifest["files"].values())
    with pytest.raises(FileExistsError):
        build_splits(output, train_count=8, dev_count=3, test_count=4, seed=11)

