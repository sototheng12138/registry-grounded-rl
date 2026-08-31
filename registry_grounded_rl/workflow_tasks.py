"""Deterministic stateful ticket workflows with hidden entity identifiers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import random
from typing import Any, Iterable, Mapping


WORKFLOW_SEMANTICS = (
    "list_projects",
    "list_tickets",
    "set_status",
    "assign_owner",
    "add_label",
)
MUTATION_SEMANTICS = ("set_status", "assign_owner", "add_label")
STATUSES = ("open", "in_progress", "resolved")


@dataclass(frozen=True, slots=True)
class ProjectRecord:
    project_id: str
    name: str

    def to_dict(self) -> dict[str, str]:
        return {"project_id": self.project_id, "name": self.name}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProjectRecord":
        return cls(project_id=str(value["project_id"]), name=str(value["name"]))


@dataclass(frozen=True, slots=True)
class TicketRecord:
    ticket_id: str
    project_id: str
    title: str
    status: str
    assignee: str
    priority: int
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"Unsupported status: {self.status!r}")
        if not 1 <= self.priority <= 4:
            raise ValueError("priority must be in [1, 4]")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "project_id": self.project_id,
            "title": self.title,
            "status": self.status,
            "assignee": self.assignee,
            "priority": self.priority,
            "labels": list(self.labels),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TicketRecord":
        return cls(
            ticket_id=str(value["ticket_id"]),
            project_id=str(value["project_id"]),
            title=str(value["title"]),
            status=str(value["status"]),
            assignee=str(value["assignee"]),
            priority=int(value["priority"]),
            labels=tuple(str(item) for item in value["labels"]),
        )


@dataclass(frozen=True, slots=True)
class WorkflowTaskSpec:
    """One end-state task; IDs are discoverable only through read tools."""

    task_id: str
    split: str
    request: str
    projects: tuple[ProjectRecord, ...]
    tickets: tuple[TicketRecord, ...]
    target_ticket_id: str
    desired_status: str
    desired_assignee: str
    desired_label: str
    missing_semantic: str
    template_family: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "dev", "test", "smoke"}:
            raise ValueError(f"Unsupported split: {self.split!r}")
        if self.desired_status not in STATUSES:
            raise ValueError(f"Unsupported desired status: {self.desired_status!r}")
        if self.missing_semantic not in MUTATION_SEMANTICS:
            raise ValueError("missing_semantic must be a required mutation")
        project_ids = [project.project_id for project in self.projects]
        ticket_ids = [ticket.ticket_id for ticket in self.tickets]
        if len(project_ids) != len(set(project_ids)):
            raise ValueError("project IDs must be unique")
        if len(ticket_ids) != len(set(ticket_ids)):
            raise ValueError("ticket IDs must be unique")
        if self.target_ticket_id not in set(ticket_ids):
            raise ValueError("target ticket is absent")
        if any(ticket.project_id not in set(project_ids) for ticket in self.tickets):
            raise ValueError("ticket references an unknown project")
        target = self.target_ticket
        if target.status == self.desired_status:
            raise ValueError("desired status must change the target")
        if target.assignee == self.desired_assignee:
            raise ValueError("desired assignee must change the target")
        if self.desired_label in target.labels:
            raise ValueError("desired label must be absent initially")

    @property
    def target_ticket(self) -> TicketRecord:
        return next(ticket for ticket in self.tickets if ticket.ticket_id == self.target_ticket_id)

    @property
    def target_project(self) -> ProjectRecord:
        target = self.target_ticket
        return next(project for project in self.projects if project.project_id == target.project_id)

    @property
    def expected_target(self) -> TicketRecord:
        return replace(
            self.target_ticket,
            status=self.desired_status,
            assignee=self.desired_assignee,
            labels=tuple(sorted((*self.target_ticket.labels, self.desired_label))),
        )

    def expected_tickets(self) -> tuple[TicketRecord, ...]:
        return tuple(
            self.expected_target if ticket.ticket_id == self.target_ticket_id else ticket
            for ticket in self.tickets
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "request": self.request,
            "projects": [project.to_dict() for project in self.projects],
            "tickets": [ticket.to_dict() for ticket in self.tickets],
            "target_ticket_id": self.target_ticket_id,
            "desired_status": self.desired_status,
            "desired_assignee": self.desired_assignee,
            "desired_label": self.desired_label,
            "missing_semantic": self.missing_semantic,
            "template_family": self.template_family,
            "expected_target": self.expected_target.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowTaskSpec":
        task = cls(
            task_id=str(value["task_id"]),
            split=str(value["split"]),
            request=str(value["request"]),
            projects=tuple(ProjectRecord.from_dict(item) for item in value["projects"]),
            tickets=tuple(TicketRecord.from_dict(item) for item in value["tickets"]),
            target_ticket_id=str(value["target_ticket_id"]),
            desired_status=str(value["desired_status"]),
            desired_assignee=str(value["desired_assignee"]),
            desired_label=str(value["desired_label"]),
            missing_semantic=str(value["missing_semantic"]),
            template_family=int(value["template_family"]),
        )
        if "expected_target" in value:
            stored = TicketRecord.from_dict(value["expected_target"])
            if stored != task.expected_target:
                raise ValueError(f"Stored expected target disagrees for {task.task_id}")
        return task


_PROJECT_NAMES = (
    "Aurora",
    "Beacon",
    "Compass",
    "Harbor",
    "Nimbus",
    "Orchid",
    "Summit",
    "Vector",
)
_TITLES = (
    "Fix login redirect",
    "Audit retry policy",
    "Refresh onboarding copy",
    "Investigate cache miss",
    "Add export validation",
    "Repair webhook parser",
    "Document rollback path",
    "Tune search ranking",
    "Resolve mobile timeout",
    "Check billing alert",
    "Improve upload recovery",
    "Verify access logs",
)
_ASSIGNEES = ("Avery", "Blair", "Casey", "Devon", "Emery", "Flynn")
_LABELS = ("backend", "customer", "data", "ops", "quality", "release")


def _stable_id(prefix: str, task_id: str, entity: str) -> str:
    digest = hashlib.sha256(f"{task_id}:{entity}".encode()).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _render_request(
    family: int,
    *,
    project: str,
    title: str,
    status: str,
    assignee: str,
    label: str,
) -> str:
    if family == 0:
        return (
            f'In project "{project}", find the ticket titled "{title}". Change its status '
            f'to "{status}", assign it to "{assignee}", and add the label "{label}". '
            "Use the registry tools and finish only after the workspace reflects all changes."
        )
    if family == 1:
        return (
            f'Please update "{title}" inside "{project}": owner = "{assignee}", '
            f'state = "{status}", label += "{label}". Discover the internal IDs with tools; '
            "do not guess them."
        )
    if family == 2:
        return (
            f'The "{project}" workspace contains an item named "{title}". Route it to '
            f'"{assignee}", mark it "{status}", and tag it "{label}". Confirm completion '
            "through the terminal action."
        )
    return (
        f'Apply this ticket workflow in "{project}" to "{title}": set status "{status}"; '
        f'set assignee "{assignee}"; attach label "{label}". Resolve project and ticket '
        "references from the live tool outputs."
    )


def generate_workflow_tasks(
    count: int,
    *,
    seed: int,
    split: str,
    template_families: Iterable[int] | None = None,
) -> tuple[WorkflowTaskSpec, ...]:
    """Generate model-independent stateful tasks with split-specific identifiers."""

    if count <= 0:
        raise ValueError("count must be positive")
    families = tuple(template_families if template_families is not None else range(4))
    if not families:
        raise ValueError("At least one template family is required")
    rng = random.Random(seed)
    tasks: list[WorkflowTaskSpec] = []
    for index in range(count):
        family = families[index % len(families)]
        task_id = f"wf-{split}-{seed}-{index:05d}"
        project_names = rng.sample(_PROJECT_NAMES, 2)
        projects = tuple(
            ProjectRecord(
                project_id=_stable_id("prj", task_id, f"project-{offset}"),
                name=name,
            )
            for offset, name in enumerate(project_names)
        )
        titles = rng.sample(_TITLES, 8)
        tickets: list[TicketRecord] = []
        for offset, title in enumerate(titles):
            project = projects[offset // 4]
            tickets.append(
                TicketRecord(
                    ticket_id=_stable_id("tkt", task_id, f"ticket-{offset}"),
                    project_id=project.project_id,
                    title=title,
                    status=STATUSES[rng.randrange(len(STATUSES))],
                    assignee=_ASSIGNEES[rng.randrange(len(_ASSIGNEES))],
                    priority=1 + (offset % 4),
                    labels=(_LABELS[rng.randrange(len(_LABELS))],),
                )
            )
        target_offset = rng.randrange(len(tickets))
        target = tickets[target_offset]
        desired_status = next(status for status in STATUSES if status != target.status)
        assignee_candidates = [name for name in _ASSIGNEES if name != target.assignee]
        desired_assignee = assignee_candidates[rng.randrange(len(assignee_candidates))]
        label_candidates = [label for label in _LABELS if label not in target.labels]
        desired_label = label_candidates[rng.randrange(len(label_candidates))]
        missing_semantic = MUTATION_SEMANTICS[index % len(MUTATION_SEMANTICS)]
        tasks.append(
            WorkflowTaskSpec(
                task_id=task_id,
                split=split,
                request=_render_request(
                    family,
                    project=next(
                        project.name for project in projects if project.project_id == target.project_id
                    ),
                    title=target.title,
                    status=desired_status,
                    assignee=desired_assignee,
                    label=desired_label,
                ),
                projects=projects,
                tickets=tuple(tickets),
                target_ticket_id=target.ticket_id,
                desired_status=desired_status,
                desired_assignee=desired_assignee,
                desired_label=desired_label,
                missing_semantic=missing_semantic,
                template_family=family,
            )
        )
    return tuple(tasks)
