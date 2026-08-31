import json

from registry_grounded_rl.pilot_analysis import (
    exact_mcnemar,
    hierarchical_paired_bootstrap,
    infer_workflow_view,
    paired_selective_harmonic_bootstrap,
    paired_task_bootstrap,
    paired_weighted_task_bootstrap,
    selective_harmonic_score,
    summarize_rollout_groups,
    wilson_interval,
)
from registry_grounded_rl.workflow_environment import RegistryWorkflowEnv
from registry_grounded_rl.workflow_tasks import generate_workflow_tasks


def test_exact_mcnemar_counts_directional_pairs() -> None:
    result = exact_mcnemar([False, False, True, True], [True, True, False, True])
    assert result["improved"] == 2
    assert result["regressed"] == 1
    assert result["discordant"] == 3
    assert result["p_value_two_sided"] == 1.0


def test_paired_task_bootstrap_preserves_observed_difference() -> None:
    left = {"a": {0: False, 1: True}, "b": {0: False, 1: False}}
    right = {"a": {0: True, 1: True}, "b": {0: True, 1: False}}
    result = paired_task_bootstrap(left, right, views=("a", "b"), samples=100, seed=3)
    assert result["tasks"] == 2
    assert result["difference"] == 0.5


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(48, 64)
    assert low < 0.75 < high


def test_weighted_bootstrap_matches_balanced_selective_difference() -> None:
    left = {
        "solvable_a": {0: False, 1: True},
        "solvable_b": {0: False, 1: False},
        "unavailable": {0: False, 1: False},
    }
    right = {
        "solvable_a": {0: True, 1: True},
        "solvable_b": {0: True, 1: False},
        "unavailable": {0: True, 1: True},
    }
    result = paired_weighted_task_bootstrap(
        left,
        right,
        view_weights={"solvable_a": 0.25, "solvable_b": 0.25, "unavailable": 0.5},
        samples=100,
        seed=3,
    )
    assert result["tasks"] == 2
    assert result["difference"] == 0.75


def test_weighted_bootstrap_rejects_zero_weights() -> None:
    left = {"a": {0: False}}
    right = {"a": {0: True}}
    try:
        paired_weighted_task_bootstrap(left, right, view_weights={"a": 0.0})
    except ValueError as error:
        assert "positive sum" in str(error)
    else:
        raise AssertionError("zero-sum weights should fail")


def test_infer_workflow_view_from_saved_initial_state() -> None:
    task = generate_workflow_tasks(1, seed=3, split="smoke")[0]
    state = RegistryWorkflowEnv(task, "opaque_alias").reset()
    row = {
        "conversations": [
            {"role": "user", "content": "system"},
            {"role": "assistant", "content": "ready"},
            {"role": "user", "content": json.dumps(state, sort_keys=True)},
        ]
    }
    assert infer_workflow_view(row, task) == "opaque_alias"


def test_selective_harmonic_score_penalizes_one_sided_collapse() -> None:
    assert selective_harmonic_score(0.5, 0.5) == 0.5
    assert selective_harmonic_score(0.0, 1.0) == 0.0


def test_paired_selective_harmonic_bootstrap_observed_difference() -> None:
    left = {"a": {0: False, 1: False}, "unavailable": {0: True, 1: True}}
    right = {"a": {0: True, 1: True}, "unavailable": {0: True, 1: True}}
    result = paired_selective_harmonic_bootstrap(
        left,
        right,
        solvable_views=("a",),
        samples=100,
        seed=3,
    )
    assert result["left_score"] == 0.0
    assert result["right_score"] == 1.0
    assert result["difference"] == 1.0


def test_hierarchical_bootstrap_keeps_training_seeds_as_outer_unit() -> None:
    left = [
        {"a": {0: False, 1: False}, "unavailable": {0: True, 1: True}},
        {"a": {0: False, 1: False}, "unavailable": {0: False, 1: False}},
    ]
    right = [
        {"a": {0: True, 1: True}, "unavailable": {0: True, 1: True}},
        {"a": {0: True, 1: True}, "unavailable": {0: False, 1: False}},
    ]
    result = hierarchical_paired_bootstrap(
        left,
        right,
        solvable_views=("a",),
        samples=100,
        seed=3,
    )
    assert result["training_seeds"] == 2
    assert result["metrics"]["primary_difference"]["difference"] == 1.0
    assert result["metrics"]["unavailable_difference"]["difference"] == 0.0


def _trajectory(item_id: int, answer: str, reward: float) -> dict:
    state = {
        "environment_schema": "registry-grounded-rl/stateful-workflow-env-v2",
        "request": "test",
    }
    return {
        "item_id": item_id,
        "reward": reward,
        "conversations": [
            {"role": "user", "content": "system"},
            {"role": "assistant", "content": "ready"},
            {"role": "user", "content": json.dumps(state)},
            {"role": "assistant", "content": answer},
        ],
    }


def test_rollout_group_summary_separates_diversity_from_reward_variance() -> None:
    rows = [
        _trajectory(1, "a", 0.0),
        _trajectory(1, "a", 0.0),
        _trajectory(2, "b", 0.0),
        _trajectory(2, "c", 1.0),
    ]
    result = summarize_rollout_groups(rows, rollout_n=2)
    assert result["groups"] == 2
    assert result["mean_unique_trajectories_per_group"] == 1.5
    assert result["fully_duplicated_groups"] == 1
    assert result["mixed_reward_groups"] == 1
    assert result["zero_reward_variance_groups"] == 1
    assert result["successes"] == 1


def test_rollout_group_summary_rejects_mixed_item_group() -> None:
    rows = [_trajectory(1, "a", 0.0), _trajectory(2, "b", 1.0)]
    try:
        summarize_rollout_groups(rows, rollout_n=2)
    except ValueError as error:
        assert "mixes item IDs" in str(error)
    else:
        raise AssertionError("mixed-item group should fail")
