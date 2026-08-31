import pytest

from registry_grounded_rl.registry import build_registry


def test_opaque_alias_changes_names_and_arguments() -> None:
    tools = build_registry("task-1", "opaque_alias")
    assert all(tool.exposed_name.startswith("unit_") for tool in tools)
    add = next(tool for tool in tools if tool.semantic == "add")
    canonical, result = add.execute({"input_a": 2, "input_b": 7})
    assert canonical == {"lhs": 2, "rhs": 7}
    assert result == 9


def test_schema_rejects_missing_or_extra_arguments() -> None:
    tool = build_registry("task-1", "original")[0]
    with pytest.raises(ValueError):
        tool.execute({"lhs": 1})
    with pytest.raises(ValueError):
        tool.execute({"lhs": 1, "rhs": 2, "extra": 3})


def test_unavailable_removes_requested_semantic() -> None:
    tools = build_registry("task-1", "unavailable", "add")
    assert "add" not in {tool.semantic for tool in tools}

