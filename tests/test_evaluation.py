import pytest

from registry_grounded_rl.environment import RegistryGroundedEnv, RegistryView
from registry_grounded_rl.evaluation import PRIMARY_VIEWS, evaluate_episodes
from registry_grounded_rl.oracle import oracle_actions, run_name_memorizer, run_oracle
from registry_grounded_rl.tasks import generate_tasks


def test_paired_metric_separates_grounded_and_name_memorizing_policies() -> None:
    tasks = generate_tasks(5, seed=4, split="smoke")
    views = [RegistryView(view) for view in PRIMARY_VIEWS]
    oracle = [run_oracle(RegistryGroundedEnv(task, view)) for task in tasks for view in views]
    memorizer = [
        run_name_memorizer(RegistryGroundedEnv(task, view)) for task in tasks for view in views
    ]
    oracle_summary = evaluate_episodes(oracle)
    memorizer_summary = evaluate_episodes(memorizer)
    assert oracle_summary["paired_registry_success_rate"] == 1.0
    assert memorizer_summary["paired_registry_success_rate"] == 0.0
    assert memorizer_summary["per_view"]["original"]["rate"] == 1.0


def test_duplicate_task_view_fails_closed() -> None:
    task = generate_tasks(1, seed=4, split="smoke")[0]
    row = run_oracle(RegistryGroundedEnv(task, RegistryView.ORIGINAL))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_episodes([row, row])


def test_malformed_model_actions_are_not_hidden_by_tool_call_metric() -> None:
    task = generate_tasks(1, seed=8, split="smoke")[0]
    env = RegistryGroundedEnv(task, RegistryView.ORIGINAL, max_steps=5)
    env.reset()
    env.step({"type": "invalid_model_output", "raw": "two calls"})
    for action in oracle_actions(env):
        env.step(action)
    row = env.trajectory()
    summary = evaluate_episodes([row])
    assert summary["malformed_actions"] == 1
    assert summary["malformed_action_rate"] > 0
