"""Reference policies used only to validate the environment and evaluator."""

from __future__ import annotations

from typing import Any

from .environment import RegistryGroundedEnv, RegistryView


def oracle_actions(env: RegistryGroundedEnv) -> tuple[dict[str, Any], ...]:
    """Translate hidden gold semantics into the current episode's public registry."""

    if env.view is RegistryView.UNAVAILABLE:
        return ({"type": "unavailable", "reason": "A required operation is not registered."},)
    tools = {tool.semantic: tool for tool in env.registry}
    actions: list[dict[str, Any]] = []
    for call in env.task.gold_calls():
        tool = tools[call["semantic"]]
        canonical = call["arguments"]
        actions.append(
            {
                "type": "tool_call",
                "name": tool.exposed_name,
                "arguments": {
                    tool.public_lhs: canonical["lhs"],
                    tool.public_rhs: canonical["rhs"],
                },
            }
        )
    actions.append({"type": "final", "answer": env.task.expected_answer})
    return tuple(actions)


def run_oracle(env: RegistryGroundedEnv) -> dict[str, Any]:
    """Execute hidden gold semantics; never use this policy for model evaluation."""

    env.reset()
    for action in oracle_actions(env):
        result = env.step(action)
        if result.terminated or result.truncated:
            break
    return env.trajectory()


def run_name_memorizer(env: RegistryGroundedEnv) -> dict[str, Any]:
    """A deliberately brittle policy that assumes canonical names and arguments."""

    env.reset()
    for call in env.task.gold_calls():
        env.step(
            {
                "type": "tool_call",
                "name": {
                    "add": "add_integers",
                    "subtract": "subtract_integers",
                    "multiply": "multiply_integers",
                    "maximum": "maximum_integer",
                }[call["semantic"]],
                "arguments": dict(call["arguments"]),
            }
        )
        if env.trajectory()["terminated"]:
            return env.trajectory()
    if not env.trajectory()["terminated"]:
        env.step({"type": "final", "answer": env.task.expected_answer})
    return env.trajectory()
