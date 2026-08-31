"""A deterministic, stateful, end-state-scored tool environment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .environment import RegistryView
from .workflow_registry import WorkflowToolSpec, build_workflow_registry
from .workflow_tasks import TicketRecord, WorkflowTaskSpec


@dataclass(frozen=True, slots=True)
class WorkflowStepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class RegistryWorkflowEnv:
    """Ticket workflow with hidden IDs, persistent mutations, and exact scoring."""

    def __init__(
        self,
        task: WorkflowTaskSpec,
        view: RegistryView | str,
        *,
        max_steps: int = 10,
    ) -> None:
        self.task = task
        self.view = RegistryView(view)
        missing = task.missing_semantic if self.view is RegistryView.UNAVAILABLE else None
        self.registry = build_workflow_registry(task.task_id, self.view.value, missing_semantic=missing)
        self.max_steps = max_steps
        if self.max_steps < 2:
            raise ValueError("max_steps must allow at least one tool call and one terminal action")
        self._tool_by_name = {tool.exposed_name: tool for tool in self.registry}
        self._records: list[dict[str, Any]] = []
        self._tickets: dict[str, dict[str, Any]] = {}
        self._terminated = False
        self._last_reward = 0.0
        self._terminal_reason: str | None = None
        self.reset()

    @staticmethod
    def _ticket_dict(ticket: TicketRecord) -> dict[str, Any]:
        value = ticket.to_dict()
        value["labels"] = sorted(value["labels"])
        return value

    def _reset_state(self) -> None:
        self._tickets = {
            ticket.ticket_id: self._ticket_dict(ticket) for ticket in self.task.tickets
        }

    def _state_snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                **self._tickets[ticket_id],
                "labels": sorted(self._tickets[ticket_id]["labels"]),
            }
            for ticket_id in sorted(self._tickets)
        ]

    def _expected_snapshot(self) -> list[dict[str, Any]]:
        return [
            self._ticket_dict(ticket)
            for ticket in sorted(self.task.expected_tickets(), key=lambda row: row.ticket_id)
        ]

    def _state_fingerprint(self) -> str:
        encoded = json.dumps(self._state_snapshot(), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def reset(self) -> dict[str, Any]:
        self._records.clear()
        self._reset_state()
        self._terminated = False
        self._last_reward = 0.0
        self._terminal_reason = None
        return {
            "environment_schema": "registry-grounded-rl/stateful-workflow-env-v2",
            "task_id": self.task.task_id,
            "request": self.task.request,
            "tools": [tool.schema() for tool in self.registry],
            "action_contract": {
                "tool_call": {"type": "tool_call", "name": "string", "arguments": "object"},
                "final": {"type": "final", "answer": "done"},
                "unavailable": {"type": "unavailable", "reason": "string"},
            },
            "max_steps": self.max_steps,
        }

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)

    def _execute(self, tool: WorkflowToolSpec, arguments: dict[str, str]) -> tuple[Any, bool]:
        semantic = tool.semantic
        if semantic == "list_projects":
            return [project.to_dict() for project in self.task.projects], False
        if semantic == "list_tickets":
            project_id = arguments["project_id"]
            if project_id not in {project.project_id for project in self.task.projects}:
                raise ValueError("unknown_project_id")
            rows = [
                {**ticket, "labels": sorted(ticket["labels"])}
                for ticket in self._tickets.values()
                if ticket["project_id"] == project_id
            ]
            rows.sort(key=lambda row: row["ticket_id"])
            return rows, False

        ticket_id = arguments["ticket_id"]
        if ticket_id not in self._tickets:
            raise ValueError("unknown_ticket_id")
        ticket = self._tickets[ticket_id]
        if semantic == "set_status":
            ticket["status"] = arguments["status"]
            return {"ticket_id": ticket_id, "status": ticket["status"], "persisted": True}, True
        if semantic == "assign_owner":
            ticket["assignee"] = arguments["assignee"]
            return {
                "ticket_id": ticket_id,
                "assignee": ticket["assignee"],
                "persisted": True,
            }, True
        if semantic == "add_label":
            if arguments["label"] not in ticket["labels"]:
                ticket["labels"].append(arguments["label"])
                ticket["labels"].sort()
            return {"ticket_id": ticket_id, "labels": ticket["labels"], "persisted": True}, True
        if semantic == "preview_status":
            return {
                "ticket_id": ticket_id,
                "preview_status": arguments["status"],
                "persisted": False,
            }, False
        if semantic == "suggest_owner":
            return {
                "ticket_id": ticket_id,
                "suggested_assignee": ticket["assignee"],
                "persisted": False,
            }, False
        if semantic == "preview_label":
            return {
                "ticket_id": ticket_id,
                "preview_label": arguments["label"],
                "persisted": False,
            }, False
        raise ValueError(f"unsupported_semantic:{semantic}")

    def _result(
        self,
        observation: dict[str, Any],
        *,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        components: dict[str, int] | None = None,
    ) -> WorkflowStepResult:
        return WorkflowStepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"reward_components": components or {}},
        )

    def _terminate(
        self,
        *,
        reason: str,
        reward: float,
        observation: dict[str, Any],
        components: dict[str, int],
        truncated: bool = False,
    ) -> WorkflowStepResult:
        self._terminated = True
        self._last_reward = reward
        self._terminal_reason = reason
        result = self._result(
            observation,
            reward=reward,
            terminated=not truncated,
            truncated=truncated,
            components=components,
        )
        result.info["terminal_reason"] = reason
        return result

    def step(self, action: Mapping[str, Any]) -> WorkflowStepResult:
        if self._terminated:
            raise RuntimeError("Episode already terminated; call reset before stepping again")
        if not isinstance(action, Mapping):
            raise TypeError("action must be a mapping")
        action_type = action.get("type")

        if action_type == "tool_call":
            name = action.get("name")
            raw_arguments = action.get("arguments")
            record: dict[str, Any] = {
                "index": len(self._records),
                "type": "tool_call",
                "name": name,
                "arguments": dict(raw_arguments)
                if isinstance(raw_arguments, Mapping)
                else raw_arguments,
                "valid": False,
                "semantic": None,
                "canonical_arguments": None,
                "result": None,
                "mutation_applied": False,
                "distractor": False,
                "error": None,
            }
            tool = self._tool_by_name.get(name) if isinstance(name, str) else None
            if tool is None:
                record["error"] = "unknown_tool"
            elif not isinstance(raw_arguments, Mapping):
                record["error"] = "arguments_not_object"
            else:
                try:
                    translated = tool.translate(raw_arguments)
                    result, mutated = self._execute(tool, translated)
                except (KeyError, TypeError, ValueError) as exc:
                    record["error"] = f"tool_error:{exc}"
                else:
                    record.update(
                        {
                            "valid": True,
                            "semantic": tool.semantic,
                            "canonical_arguments": translated,
                            "result": result,
                            "mutation_applied": mutated,
                            "distractor": tool.distractor,
                        }
                    )
            self._records.append(record)
            if len(self._records) >= self.max_steps:
                return self._terminate(
                    reason="step_budget_exhausted",
                    reward=0.0,
                    observation={"status": "truncated", "last_record": record},
                    components={"format": 1, "end_state": 0, "task": 0},
                    truncated=True,
                )
            return self._result(
                {
                    "status": "tool_result" if record["valid"] else "tool_error",
                    "name": name,
                    "result": record["result"],
                    "error": record["error"],
                    "persisted": record["mutation_applied"],
                    "steps_remaining": self.max_steps - len(self._records),
                },
                components={"format": 1, "end_state": 0, "task": 0},
            )

        if action_type == "final":
            answer = action.get("answer")
            answer_valid = isinstance(answer, str) and answer.strip().lower() == "done"
            exact_state = self._state_snapshot() == self._expected_snapshot()
            available = self.view is not RegistryView.UNAVAILABLE
            success = bool(answer_valid and exact_state and available)
            self._records.append(
                {
                    "index": len(self._records),
                    "type": "final",
                    "answer": answer,
                    "valid": answer_valid,
                }
            )
            return self._terminate(
                reason="success" if success else "incorrect_final",
                reward=float(success),
                observation={"status": "finished", "success": success},
                components={
                    "format": int(answer_valid),
                    "end_state": int(exact_state),
                    "availability": int(available),
                    "task": int(success),
                },
            )

        if action_type == "unavailable":
            reason_text = action.get("reason")
            mutation_count = sum(
                int(record.get("mutation_applied", False)) for record in self._records
            )
            unchanged = self._state_snapshot() == [
                self._ticket_dict(ticket)
                for ticket in sorted(self.task.tickets, key=lambda row: row.ticket_id)
            ]
            correct = bool(
                self.view is RegistryView.UNAVAILABLE
                and isinstance(reason_text, str)
                and reason_text.strip()
                and mutation_count == 0
                and unchanged
            )
            self._records.append(
                {
                    "index": len(self._records),
                    "type": "unavailable",
                    "reason": reason_text,
                    "valid": isinstance(reason_text, str) and bool(reason_text.strip()),
                }
            )
            return self._terminate(
                reason="correct_unavailable" if correct else "incorrect_unavailable",
                reward=float(correct),
                observation={"status": "finished", "success": correct},
                components={
                    "format": int(isinstance(reason_text, str) and bool(reason_text.strip())),
                    "no_side_effect": int(unchanged and mutation_count == 0),
                    "availability": int(correct),
                    "task": int(correct),
                },
            )

        self._records.append(
            {
                "index": len(self._records),
                "type": "invalid",
                "raw_action": dict(action),
                "valid": False,
                "error": "unknown_action_type",
            }
        )
        if len(self._records) >= self.max_steps:
            return self._terminate(
                reason="step_budget_exhausted",
                reward=0.0,
                observation={"status": "truncated", "error": "unknown_action_type"},
                components={"format": 0, "end_state": 0, "task": 0},
                truncated=True,
            )
        return self._result(
            {"status": "action_error", "error": "unknown_action_type"},
            components={"format": 0, "end_state": 0, "task": 0},
        )

    def trajectory(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "split": self.task.split,
            "view": self.view.value,
            "request": self.task.request,
            "records": [dict(record) for record in self._records],
            "reward": self._last_reward,
            "success": self._last_reward == 1.0,
            "terminal_reason": self._terminal_reason,
            "terminated": self._terminated,
            "max_steps": self.max_steps,
            "state_fingerprint": self._state_fingerprint(),
            "expected_state_fingerprint": hashlib.sha256(
                json.dumps(self._expected_snapshot(), sort_keys=True).encode()
            ).hexdigest(),
        }
