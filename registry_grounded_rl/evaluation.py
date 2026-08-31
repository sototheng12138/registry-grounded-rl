"""Trajectory-level and paired-registry evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


PRIMARY_VIEWS = (
    "original",
    "order",
    "schema_surface",
    "opaque_alias",
    "hard_distractor",
)


def evaluate_episodes(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in episodes]
    if not rows:
        raise ValueError("At least one episode is required")
    seen: set[tuple[str, str]] = set()
    by_task: dict[str, dict[str, bool]] = defaultdict(dict)
    view_success: Counter[str] = Counter()
    view_total: Counter[str] = Counter()
    terminal_reasons: Counter[str] = Counter()
    actions = 0
    tool_calls = 0
    invalid_tool_calls = 0
    malformed_actions = 0
    distractor_calls = 0
    mutation_calls = 0
    successful_actions = 0
    successful_episodes = 0
    unavailable_tool_calls = 0
    unavailable_mutations = 0
    unavailable_episodes_with_actions = 0

    for row in rows:
        task_id = str(row["task_id"])
        view = str(row["view"])
        key = (task_id, view)
        if key in seen:
            raise ValueError(f"Duplicate task/view episode: {key}")
        seen.add(key)
        success = bool(row.get("success", False))
        by_task[task_id][view] = success
        view_total[view] += 1
        view_success[view] += int(success)
        terminal_reasons[str(row.get("terminal_reason"))] += 1
        row_action_count = 0
        row_unavailable_tool_calls = 0
        for record in row.get("records", []):
            if not isinstance(record, Mapping):
                continue
            actions += 1
            row_action_count += 1
            if record.get("type") == "invalid":
                malformed_actions += 1
            if record.get("type") == "tool_call":
                tool_calls += 1
                invalid_tool_calls += int(not record.get("valid", False))
                distractor_calls += int(record.get("distractor", False))
                mutation_calls += int(record.get("mutation_applied", False))
                if view == "unavailable":
                    unavailable_tool_calls += 1
                    row_unavailable_tool_calls += 1
                    unavailable_mutations += int(record.get("mutation_applied", False))
        if success:
            successful_episodes += 1
            successful_actions += row_action_count
        if view == "unavailable" and row_unavailable_tool_calls:
            unavailable_episodes_with_actions += 1

    complete_primary = {
        task_id: all(view in views for view in PRIMARY_VIEWS)
        for task_id, views in by_task.items()
    }
    eligible = [task_id for task_id, complete in complete_primary.items() if complete]
    paired_success = sum(
        all(by_task[task_id][view] for view in PRIMARY_VIEWS) for task_id in eligible
    )
    per_view = {
        view: {
            "success": view_success[view],
            "total": view_total[view],
            "rate": view_success[view] / view_total[view],
        }
        for view in sorted(view_total)
    }
    return {
        "episodes": len(rows),
        "tasks": len(by_task),
        "primary_complete_tasks": len(eligible),
        "paired_registry_success": paired_success,
        "paired_registry_total": len(eligible),
        "paired_registry_success_rate": paired_success / len(eligible) if eligible else None,
        "per_view": per_view,
        "actions": actions,
        "malformed_actions": malformed_actions,
        "malformed_action_rate": malformed_actions / actions if actions else 0.0,
        "tool_calls": tool_calls,
        "invalid_tool_calls": invalid_tool_calls,
        "invalid_tool_call_rate": invalid_tool_calls / tool_calls if tool_calls else 0.0,
        "distractor_calls": distractor_calls,
        "mutation_calls": mutation_calls,
        "mean_actions_per_success": (
            successful_actions / successful_episodes if successful_episodes else None
        ),
        "unavailable_tool_calls": unavailable_tool_calls,
        "unavailable_mutations": unavailable_mutations,
        "unavailable_over_action_episodes": unavailable_episodes_with_actions,
        "unavailable_over_action_rate": (
            unavailable_episodes_with_actions / view_total["unavailable"]
            if view_total["unavailable"]
            else None
        ),
        "terminal_reasons": dict(sorted(terminal_reasons.items())),
    }
