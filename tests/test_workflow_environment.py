from registry_grounded_rl.environment import RegistryView
from registry_grounded_rl.evaluation import evaluate_episodes
from registry_grounded_rl.workflow_environment import RegistryWorkflowEnv
from registry_grounded_rl.workflow_oracle import (
    run_workflow_name_memorizer,
    run_workflow_oracle,
    workflow_oracle_actions,
)
from registry_grounded_rl.workflow_tasks import generate_workflow_tasks


def _task():
    return generate_workflow_tasks(1, seed=1701, split="smoke")[0]


def test_oracle_succeeds_in_every_registry_view() -> None:
    rows = [run_workflow_oracle(RegistryWorkflowEnv(_task(), view)) for view in RegistryView]
    summary = evaluate_episodes(rows)
    assert summary["episodes"] == 6
    assert all(row["success"] for row in rows)
    assert summary["paired_registry_success_rate"] == 1.0


def test_canonical_name_memorizer_fails_on_aliases_and_unavailable() -> None:
    rows = [
        run_workflow_name_memorizer(RegistryWorkflowEnv(_task(), view))
        for view in RegistryView
    ]
    by_view = {row["view"]: row["success"] for row in rows}
    assert by_view["original"]
    assert by_view["order"]
    assert by_view["hard_distractor"]
    assert not by_view["schema_surface"]
    assert not by_view["opaque_alias"]
    assert not by_view["unavailable"]


def test_partial_completion_receives_zero_reward() -> None:
    env = RegistryWorkflowEnv(_task(), "original")
    env.reset()
    for action in workflow_oracle_actions(env)[:-2]:
        env.step(action)
    result = env.step({"type": "final", "answer": "done"})
    assert result.reward == 0.0
    assert result.info["reward_components"]["end_state"] == 0


def test_unrelated_side_effect_invalidates_exact_end_state() -> None:
    task = _task()
    env = RegistryWorkflowEnv(task, "original", max_steps=12)
    env.reset()
    unrelated = next(ticket for ticket in task.tickets if ticket.ticket_id != task.target_ticket_id)
    changed_status = next(status for status in ("open", "in_progress", "resolved") if status != unrelated.status)
    env.step(
        {
            "type": "tool_call",
            "name": "set_ticket_status",
            "arguments": {"ticket_id": unrelated.ticket_id, "status": changed_status},
        }
    )
    for action in workflow_oracle_actions(env)[2:]:
        result = env.step(action)
    assert result.reward == 0.0
    assert result.info["reward_components"]["end_state"] == 0


def test_preview_distractor_does_not_mutate_state() -> None:
    task = _task()
    env = RegistryWorkflowEnv(task, "hard_distractor")
    before = env.trajectory()["state_fingerprint"]
    result = env.step(
        {
            "type": "tool_call",
            "name": "preview_ticket_status",
            "arguments": {"ticket_id": task.target_ticket_id, "status": task.desired_status},
        }
    )
    assert result.observation["persisted"] is False
    assert env.trajectory()["state_fingerprint"] == before


def test_unavailable_requires_no_side_effects() -> None:
    task = _task()
    env = RegistryWorkflowEnv(task, "unavailable")
    available_mutation = next(
        tool for tool in env.registry if tool.semantic in {"set_status", "assign_owner", "add_label"}
    )
    canonical_values = {
        "ticket_id": task.target_ticket_id,
        "status": task.desired_status,
        "assignee": task.desired_assignee,
        "label": task.desired_label,
    }
    env.step(
        {
            "type": "tool_call",
            "name": available_mutation.exposed_name,
            "arguments": {
                public: canonical_values[canonical]
                for canonical, public in available_mutation.argument_aliases
            },
        }
    )
    result = env.step({"type": "unavailable", "reason": "A capability is absent."})
    assert result.reward == 0.0
    assert result.info["reward_components"]["no_side_effect"] == 0


def test_unavailable_replaces_missing_mutation_without_cardinality_shortcut() -> None:
    task = _task()
    original = RegistryWorkflowEnv(task, "original")
    unavailable = RegistryWorkflowEnv(task, "unavailable")
    assert len(original.registry) == len(unavailable.registry) == 5
    assert task.missing_semantic not in {tool.semantic for tool in unavailable.registry}
    assert sum(tool.distractor for tool in unavailable.registry) == 1
