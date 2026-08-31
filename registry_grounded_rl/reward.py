"""Harness-independent replay reward for generated action trajectories."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .environment import RegistryGroundedEnv, RegistryView
from .tasks import TaskSpec


def replay_and_score(
    task: TaskSpec,
    view: RegistryView | str,
    actions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay model actions and return the same binary reward used by evaluation."""

    env = RegistryGroundedEnv(task, view)
    env.reset()
    for action in actions:
        result = env.step(action)
        if result.terminated or result.truncated:
            break
    trajectory = env.trajectory()
    return {
        "reward": float(trajectory["reward"]),
        "success": bool(trajectory["success"]),
        "terminal_reason": trajectory["terminal_reason"],
        "trajectory": trajectory,
    }

