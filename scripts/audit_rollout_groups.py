#!/usr/bin/env python3
"""Audit GRPO rollout diversity and within-group reward variance from saved trajectories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from registry_grounded_rl.pilot_analysis import summarize_rollout_groups
from registry_grounded_rl.workflow_dataset import load_workflow_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    run_dir = arguments.run_dir.expanduser().resolve(strict=True)
    task_file = arguments.task_file.expanduser().resolve(strict=True)
    tasks = dict(enumerate(load_workflow_tasks(task_file)))
    steps = {}
    step_dirs = sorted(
        (run_dir / "trajectories").glob("step*"),
        key=lambda path: int(path.name.removeprefix("step")),
    )
    for step_dir in step_dirs:
        if not step_dir.is_dir():
            continue
        rows = []
        for trajectory_file in sorted(step_dir.rglob("*.json")):
            payload = json.loads(trajectory_file.read_text())
            if not isinstance(payload, list):
                raise ValueError(f"trajectory payload is not a list: {trajectory_file}")
            rows.extend(payload)
        steps[step_dir.name] = summarize_rollout_groups(
            rows,
            rollout_n=arguments.rollout_n,
            tasks=tasks,
        )
    if not steps:
        raise ValueError(f"no step trajectories found in {run_dir}")
    result = {
        "schema_version": "registry-grounded-rl/rollout-group-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "task_file": str(task_file),
        "rollout_n": arguments.rollout_n,
        "steps": steps,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
