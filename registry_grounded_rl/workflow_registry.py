"""Episode-local tool surfaces for deterministic stateful workflows."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Mapping

from .workflow_tasks import STATUSES, WORKFLOW_SEMANTICS


CANONICAL_WORKFLOW_NAMES = {
    "list_projects": "list_projects",
    "list_tickets": "list_tickets",
    "set_status": "set_ticket_status",
    "assign_owner": "assign_ticket",
    "add_label": "add_ticket_label",
}
WORKFLOW_DESCRIPTIONS = {
    "list_projects": "List all projects and return each project's live internal ID and name.",
    "list_tickets": "List the tickets in one project, including live ticket IDs and fields.",
    "set_status": "Persistently change one ticket's status.",
    "assign_owner": "Persistently change one ticket's assignee.",
    "add_label": "Persistently add one label to a ticket without removing existing labels.",
    "preview_status": "Preview a status change without modifying the workspace.",
    "suggest_owner": "Suggest an assignee without modifying the workspace.",
    "preview_label": "Preview a label addition without modifying the workspace.",
}
CANONICAL_ARGUMENTS = {
    "list_projects": (),
    "list_tickets": ("project_id",),
    "set_status": ("ticket_id", "status"),
    "assign_owner": ("ticket_id", "assignee"),
    "add_label": ("ticket_id", "label"),
    "preview_status": ("ticket_id", "status"),
    "suggest_owner": ("ticket_id",),
    "preview_label": ("ticket_id", "label"),
}


@dataclass(frozen=True, slots=True)
class WorkflowToolSpec:
    semantic: str
    exposed_name: str
    argument_aliases: tuple[tuple[str, str], ...]
    distractor: bool = False
    reverse_schema_order: bool = False

    @property
    def canonical_arguments(self) -> tuple[str, ...]:
        return tuple(canonical for canonical, _ in self.argument_aliases)

    @property
    def public_arguments(self) -> tuple[str, ...]:
        return tuple(public for _, public in self.argument_aliases)

    def schema(self) -> dict[str, Any]:
        aliases = list(self.argument_aliases)
        if self.reverse_schema_order:
            aliases.reverse()
        properties: dict[str, Any] = {}
        for canonical, public in aliases:
            value: dict[str, Any] = {
                "type": "string",
                "description": {
                    "project_id": "A project ID returned by the project-listing tool.",
                    "ticket_id": "A ticket ID returned by the ticket-listing tool.",
                    "status": "The new persistent status.",
                    "assignee": "The exact requested assignee name.",
                    "label": "The exact label to add.",
                }[canonical],
            }
            if canonical == "status":
                value["enum"] = list(STATUSES)
            properties[public] = value
        return {
            "type": "function",
            "function": {
                "name": self.exposed_name,
                "description": WORKFLOW_DESCRIPTIONS[self.semantic],
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(self.public_arguments),
                    "properties": properties,
                },
            },
        }

    def translate(self, arguments: Mapping[str, Any]) -> dict[str, str]:
        if set(arguments) != set(self.public_arguments):
            raise ValueError(
                f"expected arguments {sorted(self.public_arguments)}, got {sorted(arguments)}"
            )
        translated: dict[str, str] = {}
        for canonical, public in self.argument_aliases:
            value = arguments[public]
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{public} must be a non-empty string")
            translated[canonical] = value
        if translated.get("status") not in {None, *STATUSES}:
            raise ValueError(f"unsupported status: {translated['status']!r}")
        return translated


def _stable_name(task_id: str, semantic: str) -> str:
    digest = hashlib.sha256(f"workflow:{task_id}:{semantic}".encode()).hexdigest()[:10]
    return f"unit_{digest}"


def _aliases(semantic: str, view: str) -> tuple[tuple[str, str], ...]:
    canonical = CANONICAL_ARGUMENTS[semantic]
    if view == "opaque_alias":
        return tuple((name, f"arg_{index}") for index, name in enumerate(canonical))
    if view == "schema_surface":
        public = {
            "project_id": "workspace_ref",
            "ticket_id": "item_ref",
            "status": "new_state",
            "assignee": "new_owner",
            "label": "tag_value",
        }
        return tuple((name, public[name]) for name in canonical)
    return tuple((name, name) for name in canonical)


def build_workflow_registry(
    task_id: str,
    view: str,
    *,
    missing_semantic: str | None = None,
) -> tuple[WorkflowToolSpec, ...]:
    """Build a semantic-equivalent registry, except in the explicit unavailable view."""

    valid_views = {
        "original",
        "order",
        "schema_surface",
        "opaque_alias",
        "hard_distractor",
        "unavailable",
    }
    if view not in valid_views:
        raise ValueError(f"Unsupported view: {view!r}")
    semantics = list(WORKFLOW_SEMANTICS)
    if view == "unavailable":
        if missing_semantic not in semantics:
            raise ValueError("unavailable view requires a registered missing semantic")
        semantics.remove(str(missing_semantic))

    tools = [
        WorkflowToolSpec(
            semantic=semantic,
            exposed_name=(
                _stable_name(task_id, semantic)
                if view == "opaque_alias"
                else CANONICAL_WORKFLOW_NAMES[semantic]
            ),
            argument_aliases=_aliases(semantic, view),
            reverse_schema_order=view == "schema_surface",
        )
        for semantic in semantics
    ]
    if view == "unavailable":
        replacement = {
            "set_status": ("preview_status", "preview_ticket_status"),
            "assign_owner": ("suggest_owner", "suggest_ticket_assignee"),
            "add_label": ("preview_label", "preview_ticket_label"),
        }[str(missing_semantic)]
        semantic, exposed_name = replacement
        tools.append(
            WorkflowToolSpec(
                semantic=semantic,
                exposed_name=exposed_name,
                argument_aliases=_aliases(semantic, view),
                distractor=True,
            )
        )
    if view == "hard_distractor":
        tools.extend(
            [
                WorkflowToolSpec(
                    semantic="preview_status",
                    exposed_name="preview_ticket_status",
                    argument_aliases=_aliases("preview_status", view),
                    distractor=True,
                ),
                WorkflowToolSpec(
                    semantic="suggest_owner",
                    exposed_name="suggest_ticket_assignee",
                    argument_aliases=_aliases("suggest_owner", view),
                    distractor=True,
                ),
                WorkflowToolSpec(
                    semantic="preview_label",
                    exposed_name="preview_ticket_label",
                    argument_aliases=_aliases("preview_label", view),
                    distractor=True,
                ),
            ]
        )
    if view in {"order", "hard_distractor"}:
        digest = hashlib.sha256(f"workflow:{task_id}:{view}".encode()).hexdigest()[:16]
        random.Random(int(digest, 16)).shuffle(tools)
    return tuple(tools)
