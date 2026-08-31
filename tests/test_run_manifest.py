import json

import pytest

from registry_grounded_rl.run_manifest import main


def test_write_once_run_manifest(tmp_path) -> None:
    project = tmp_path / "project"
    package = project / "registry_grounded_rl"
    scripts = project / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    (package / "module.py").write_text("VALUE = 1\n")
    (scripts / "train.sh").write_text("#!/bin/sh\n")
    agentgym = tmp_path / "agentgym"
    agentgym.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n")
    (model / "model.safetensors").write_bytes(b"frozen-test-weights")
    task = tmp_path / "train.jsonl"
    index = tmp_path / "index.jsonl"
    task.write_text("{}\n")
    index.write_text("{}\n")
    output = tmp_path / "run" / "run_manifest.json"
    argv = [
        "--output",
        str(output),
        "--project-root",
        str(project),
        "--agentgym-root",
        str(agentgym),
        "--model-path",
        str(model),
        "--task-file",
        str(task),
        "--index-file",
        str(index),
        "--environment-address",
        f"workflow://{task}?view=original",
        "--arm",
        "original",
        "--benchmark",
        "workflow",
        "--run-seed",
        "1701",
        "--n-gpus",
        "2",
        "--train-batch-size",
        "16",
        "--rollout-n",
        "6",
        "--total-epochs",
        "1",
    ]
    assert main(argv) == 0
    value = json.loads(output.read_text())
    assert value["task_file"]["sha256"]
    assert value["source"]["code_sha256"]
    assert value["training"]["rollout_n"] == 6
    assert value["training"]["data_seed"] == 1701
    assert value["training"]["rollout_seed"] == 1701
    assert value["training"]["actor_lr"] == 1e-6
    assert value["training"]["entropy_coeff"] == 0.001
    assert value["training"]["kl_loss_coef"] == 0.001
    assert value["model"]["weight_files"] == ["model.safetensors"]
    assert value["model"]["weights_sha256"]
    with pytest.raises(FileExistsError):
        main(argv)
