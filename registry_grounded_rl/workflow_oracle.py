"""Reference and deliberately brittle policies for workflow validation."""

from __future__ import annotations

from typing import Any

from .environment import RegistryView
from .workflow_environment import RegistryWorkflowEnv
from .workflow_registry import CANONICAL_WORKFLOW_NAMES, WorkflowToolSpec


def _call(tool: WorkflowToolSpec, canonical: dict[str, str]) -> dict[str, Any]:
    aliases = dict(tool.argument_aliases)
    return {
        "type": "tool_call",
        "name": tool.exposed_name,
        "arguments": {aliases[name]: value for name, value in canonical.items()},
    }


def workflow_oracle_actions(env: RegistryWorkflowEnv) -> tuple[dict[str, Any], ...]:
    if env.view is RegistryView.UNAVAILABLE:
        return (
            {
                "type": "unavailable",
                "reason": f"The required {env.task.missing_semantic} capability is not registered.",
            },
        )
    tools = {tool.semantic: tool for tool in env.registry if not tool.distractor}
    task = env.task
    return (
        _call(tools["list_projects"], {}),
        _call(tools["list_tickets"], {"project_id": task.target_project.project_id}),
        _call(
            tools["set_status"],
            {"ticket_id": task.target_ticket_id, "status": task.desired_status},
        ),
        _call(
            tools["assign_owner"],
            {"ticket_id": task.target_ticket_id, "assignee": task.desired_assignee},
        ),
        _call(
            tools["add_label"],
            {"ticket_id": task.target_ticket_id, "label": task.desired_label},
        ),
        {"type": "final", "answer": "done"},
    )


def run_workflow_oracle(env: RegistryWorkflowEnv) -> dict[str, Any]:
    env.reset()
    for action in workflow_oracle_actions(env):
        result = env.step(action)
        if result.terminated or result.truncated:
            break
    return env.trajectory()


def run_workflow_name_memorizer(env: RegistryWorkflowEnv) -> dict[str, Any]:
    """Assume canonical names and keys, demonstrating the intended shortcut failure."""

    task = env.task
    env.reset()
    calls = (
        ("list_projects", {}),
        ("list_tickets", {"project_id": task.target_project.project_id}),
        (
            "set_status",
            {"ticket_id": task.target_ticket_id, "status": task.desired_status},
        ),
        (
            "assign_owner",
            {"ticket_id": task.target_ticket_id, "assignee": task.desired_assignee},
        ),
        (
            "add_label",
            {"ticket_id": task.target_ticket_id, "label": task.desired_label},
        ),
    )
    for semantic, arguments in calls:
        result = env.step(
            {
                "type": "tool_call",
                "name": CANONICAL_WORKFLOW_NAMES[semantic],
                "arguments": arguments,
            }
        )
        if result.terminated or result.truncated:
            return env.trajectory()
    if not env.trajectory()["terminated"]:
        env.step({"type": "final", "answer": "done"})
    return env.trajectory()
