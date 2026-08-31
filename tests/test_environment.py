import pytest

from registry_grounded_rl.environment import RegistryGroundedEnv, RegistryView
from registry_grounded_rl.oracle import run_name_memorizer, run_oracle
from registry_grounded_rl.tasks import PREVIOUS, OperationStep, TaskSpec


@pytest.fixture
def task() -> TaskSpec:
    return TaskSpec(
        task_id="rg-smoke-env",
        split="smoke",
        request="Add 3 and 4, then multiply by 5.",
        steps=(OperationStep("add", 3, 4), OperationStep("multiply", PREVIOUS, 5)),
        template_family=0,
    )


@pytest.mark.parametrize("view", list(RegistryView))
def test_oracle_succeeds_in_every_registry_view(task: TaskSpec, view: RegistryView) -> None:
    trajectory = run_oracle(RegistryGroundedEnv(task, view))
    assert trajectory["success"] is True
    assert trajectory["reward"] == 1.0


def test_name_memorizer_fails_on_opaque_alias(task: TaskSpec) -> None:
    trajectory = run_name_memorizer(RegistryGroundedEnv(task, RegistryView.OPAQUE_ALIAS))
    assert trajectory["success"] is False
    assert any(record.get("error") == "unknown_tool" for record in trajectory["records"])


def test_correct_answer_without_verified_trace_gets_zero(task: TaskSpec) -> None:
    env = RegistryGroundedEnv(task, RegistryView.ORIGINAL)
    env.reset()
    result = env.step({"type": "final", "answer": 35})
    assert result.reward == 0.0
    assert result.info["reward_components"]["answer"] == 1
    assert result.info["reward_components"]["trace"] == 0


def test_wrong_tool_then_correct_trace_does_not_reward(task: TaskSpec) -> None:
    env = RegistryGroundedEnv(task, RegistryView.ORIGINAL, max_steps=7)
    env.reset()
    env.step(
        {"type": "tool_call", "name": "maximum_integer", "arguments": {"lhs": 3, "rhs": 4}}
    )
    env.step({"type": "tool_call", "name": "add_integers", "arguments": {"lhs": 3, "rhs": 4}})
    env.step(
        {
            "type": "tool_call",
            "name": "multiply_integers",
            "arguments": {"lhs": 7, "rhs": 5},
        }
    )
    result = env.step({"type": "final", "answer": 35})
    assert result.reward == 0.0


def test_unavailable_requires_no_prior_tool_call(task: TaskSpec) -> None:
    env = RegistryGroundedEnv(task, RegistryView.UNAVAILABLE)
    env.reset()
    result = env.step({"type": "unavailable", "reason": "add is absent"})
    assert result.reward == 1.0

