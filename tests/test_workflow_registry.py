from registry_grounded_rl.workflow_registry import build_workflow_registry


def test_opaque_registry_changes_names_and_argument_keys() -> None:
    tools = build_workflow_registry("task-1", "opaque_alias")
    assert all(tool.exposed_name.startswith("unit_") for tool in tools)
    mutation = next(tool for tool in tools if tool.semantic == "set_status")
    assert mutation.public_arguments == ("arg_0", "arg_1")
    assert mutation.translate({"arg_0": "tkt_x", "arg_1": "resolved"}) == {
        "ticket_id": "tkt_x",
        "status": "resolved",
    }


def test_schema_surface_is_lossless_but_not_canonical() -> None:
    tools = build_workflow_registry("task-1", "schema_surface")
    mutation = next(tool for tool in tools if tool.semantic == "assign_owner")
    assert mutation.exposed_name == "assign_ticket"
    assert mutation.public_arguments == ("item_ref", "new_owner")
    assert mutation.translate({"item_ref": "tkt_x", "new_owner": "Avery"}) == {
        "ticket_id": "tkt_x",
        "assignee": "Avery",
    }


def test_distractor_and_unavailable_views_change_only_intended_capability() -> None:
    hard = build_workflow_registry("task-1", "hard_distractor")
    assert sum(tool.distractor for tool in hard) == 3
    unavailable = build_workflow_registry(
        "task-1", "unavailable", missing_semantic="set_status"
    )
    assert len(unavailable) == 5
    assert "set_status" not in {tool.semantic for tool in unavailable}
    assert {tool.semantic for tool in unavailable} == {
        "list_projects",
        "list_tickets",
        "assign_owner",
        "add_label",
        "preview_status",
    }
    replacement = next(tool for tool in unavailable if tool.semantic == "preview_status")
    assert replacement.distractor
