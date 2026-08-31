import json

from registry_grounded_rl.agentgym_compat import (
    balanced_capability_orbit_view,
    balanced_orbit_view,
    grouped_capability_orbit_view,
    selective_capability_orbit_view,
    stratified_capability_orbit_view,
    stratified_solvable_orbit_view,
)
from registry_grounded_rl.workflow_agentgym import RegistryWorkflowAgentGymClient
from registry_grounded_rl.workflow_oracle import workflow_oracle_actions
from registry_grounded_rl.workflow_tasks import generate_workflow_tasks


def _write_task(path) -> None:
    task = generate_workflow_tasks(1, seed=1701, split="smoke")[0]
    path.write_text(json.dumps(task.to_dict()) + "\n")


def _render(action: dict) -> str:
    payload = {key: value for key, value in action.items() if key != "type"}
    return f"<{action['type']}>{json.dumps(payload)}</{action['type']}>"


def test_balanced_orbit_covers_all_primary_views_per_five_clients() -> None:
    views = {
        balanced_orbit_view(seed=1701, item_id=7, client_serial=serial, salt="workflow")
        for serial in range(5)
    }
    assert {view.value for view in views} == {
        "original",
        "order",
        "schema_surface",
        "opaque_alias",
        "hard_distractor",
    }


def test_capability_orbit_covers_solvable_and_unavailable_per_six_clients() -> None:
    views = {
        balanced_capability_orbit_view(
            seed=1701, item_id=7, client_serial=serial, salt="workflow-capability"
        )
        for serial in range(6)
    }
    assert {view.value for view in views} == {
        "original",
        "order",
        "schema_surface",
        "opaque_alias",
        "hard_distractor",
        "unavailable",
    }


def test_grouped_capability_orbit_holds_state_within_group() -> None:
    views = [
        grouped_capability_orbit_view(
            seed=1701,
            item_id=7,
            occurrence=occurrence,
            group_size=6,
        )
        for occurrence in range(36)
    ]
    assert all(len(set(views[start : start + 6])) == 1 for start in range(0, 36, 6))
    assert {view.value for view in views} == {
        "original",
        "order",
        "schema_surface",
        "opaque_alias",
        "hard_distractor",
        "unavailable",
    }


def test_stratified_capability_orbit_is_grouped_and_globally_balanced() -> None:
    views = [
        stratified_capability_orbit_view(
            seed=1701,
            occurrence=occurrence,
            group_size=6,
        )
        for occurrence in range(36)
    ]
    assert all(len(set(views[start : start + 6])) == 1 for start in range(0, 36, 6))
    assert len(set(views)) == 6


def test_selective_capability_orbit_matches_balanced_objective() -> None:
    views = [
        selective_capability_orbit_view(
            seed=1701,
            occurrence=occurrence,
            group_size=6,
        )
        for occurrence in range(60)
    ]
    groups = [views[start : start + 6] for start in range(0, 60, 6)]
    assert all(len(set(group)) == 1 for group in groups)
    assert sum(group[0].value == "unavailable" for group in groups) == 5
    assert {group[0].value for group in groups[::2]} == {
        "original",
        "order",
        "schema_surface",
        "opaque_alias",
        "hard_distractor",
    }


def test_stratified_solvable_orbit_is_grouped_and_covers_primary_views() -> None:
    views = [
        stratified_solvable_orbit_view(
            seed=1701,
            occurrence=occurrence,
            group_size=6,
        )
        for occurrence in range(30)
    ]
    groups = [views[start : start + 6] for start in range(0, 30, 6)]
    assert all(len(set(group)) == 1 for group in groups)
    assert {group[0].value for group in groups} == {
        "original",
        "order",
        "schema_surface",
        "opaque_alias",
        "hard_distractor",
    }


def test_workflow_agentgym_client_executes_multiturn_episode(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    _write_task(path)
    client = RegistryWorkflowAgentGymClient(
        f"workflow://{path}?view=original&seed=1701", data_len=1
    )
    client.reset(0)
    assert "exact terminal workspace state" in client.conversation_start[0]["value"]
    assert "list_projects" in client.observe()
    assert client.env is not None
    for action in workflow_oracle_actions(client.env):
        output = client.step(_render(action))
    assert output.done
    assert output.reward == 1.0


def test_workflow_agentgym_client_accepts_contract_json(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    _write_task(path)
    client = RegistryWorkflowAgentGymClient(
        f"workflow://{path}?view=original&seed=1701", data_len=1
    )
    client.reset(0)
    output = client.step('{"type":"tool_call","name":"list_projects","arguments":{}}')
    assert not output.done
    assert "tool_result" in output.state


def test_grouped_client_assigns_one_view_to_all_six_rollouts(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    _write_task(path)
    address = f"workflow://{path}?view=grouped_capability_orbit&seed=99173&group_size=6"
    clients = [RegistryWorkflowAgentGymClient(address, data_len=1) for _ in range(12)]
    for client in clients:
        client.reset(0)
    views = [client.env.view for client in clients if client.env is not None]
    assert len(set(views[:6])) == 1
    assert len(set(views[6:])) == 1
    assert views[0] != views[6]
