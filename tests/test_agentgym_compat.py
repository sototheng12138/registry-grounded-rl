import json
from pathlib import Path

import pytest

from registry_grounded_rl.agentgym_compat import (
    RegistryGroundedAgentGymClient,
    parse_inproc_address,
)
from registry_grounded_rl.agentgym_data import build_agentgym_index, build_numeric_item_splits
from registry_grounded_rl.dataset import build_splits
from registry_grounded_rl.oracle import oracle_actions


def _task_file(tmp_path: Path) -> Path:
    output = tmp_path / "tasks"
    build_splits(output, train_count=3, dev_count=1, test_count=1, seed=19)
    return output / "train.jsonl"


def test_parse_inproc_address(tmp_path: Path) -> None:
    tasks = _task_file(tmp_path)
    path, view, seed = parse_inproc_address(f"inproc://{tasks}?view=orbit&seed=7")
    assert path == tasks.resolve()
    assert view == "orbit"
    assert seed == 7


def test_agentgym_client_executes_oracle_trace(tmp_path: Path) -> None:
    tasks = _task_file(tmp_path)
    client = RegistryGroundedAgentGymClient(
        f"inproc://{tasks}?view=original&seed=7", data_len=1
    )
    initial = client.reset(0)
    assert json.loads(client.observe())["task_id"] == initial["task_id"]
    assert client.env is not None
    actions = oracle_actions(client.env)
    output = None
    for action in actions:
        if action["type"] == "tool_call":
            text = (
                '<tool_call>{"name":'
                + json.dumps(action["name"])
                + ',"arguments":'
                + json.dumps(action["arguments"])
                + "}</tool_call>"
            )
        else:
            text = f'<final>{{"answer":{action["answer"]}}}</final>'
        output = client.step(text)
    assert output is not None
    assert output.done is True
    assert output.reward == 1.0


def test_agentgym_index_is_write_once_and_hashed(tmp_path: Path) -> None:
    tasks = _task_file(tmp_path)
    output = tmp_path / "agentgym.jsonl"
    result = build_agentgym_index(tasks, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["rows"] == 3
    assert len(result["sha256"]) == 64
    assert rows[0]["item_id"] == "registrygrounded_0"
    with pytest.raises(FileExistsError):
        build_agentgym_index(tasks, output)


def test_numeric_item_splits_are_disjoint_and_complete(tmp_path: Path) -> None:
    output = tmp_path / "todo"
    manifest = build_numeric_item_splits(
        output, prefix="registrytodo", count=20, train_count=12, dev_count=3, seed=5
    )
    split_ids = []
    for split in ("train", "dev", "test"):
        rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text().splitlines()]
        split_ids.append({int(row["item_id"].rsplit("_", 1)[1]) for row in rows})
    assert not (split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2])
    assert len(set().union(*split_ids)) == 20
    assert manifest["model_outputs_used"] is False
