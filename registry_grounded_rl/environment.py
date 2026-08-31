"""A small, stateful, executable environment for multi-turn tool agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .registry import build_registry
from .tasks import TaskSpec


class RegistryView(str, Enum):
    ORIGINAL = "original"
    ORDER = "order"
    SCHEMA_SURFACE = "schema_surface"
    OPAQUE_ALIAS = "opaque_alias"
    HARD_DISTRACTOR = "hard_distractor"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class RegistryGroundedEnv:
    """Gym-like environment with JSON actions and deterministic end-state reward."""

    def __init__(self, task: TaskSpec, view: RegistryView | str, *, max_steps: int | None = None):
        self.task = task
        self.view = RegistryView(view)
        self.missing_semantic = task.steps[0].semantic if self.view is RegistryView.UNAVAILABLE else None
        self.registry = build_registry(task.task_id, self.view.value, self.missing_semantic)
        self.max_steps = max_steps if max_steps is not None else len(task.steps) + 3
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._tool_by_name = {tool.exposed_name: tool for tool in self.registry}
        self._records: list[dict[str, Any]] = []
        self._terminated = False
        self._last_reward = 0.0
        self._terminal_reason: str | None = None

    def reset(self) -> dict[str, Any]:
        self._records.clear()
        self._terminated = False
        self._last_reward = 0.0
        self._terminal_reason = None
        return {
            "task_id": self.task.task_id,
            "request": self.task.request,
            "tools": [tool.schema() for tool in self.registry],
            "action_contract": {
                "tool_call": {"type": "tool_call", "name": "string", "arguments": "object"},
                "final": {"type": "final", "answer": "integer"},
                "unavailable": {"type": "unavailable", "reason": "string"},
            },
            "max_steps": self.max_steps,
        }

    def available_actions(self) -> dict[str, Any]:
        return {
            "tools": [tool.schema() for tool in self.registry],
            "terminal_actions": ["final", "unavailable"],
        }

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record) for record in self._records)

    def _gold_trace_matches(self) -> bool:
        calls = [record for record in self._records if record["type"] == "tool_call"]
        if any(not record["valid"] for record in calls):
            return False
        compact = [
            {
                "semantic": record["semantic"],
                "arguments": record["canonical_arguments"],
                "result": record["result"],
            }
            for record in calls
        ]
        return compact == list(self.task.gold_calls())

    @staticmethod
    def _integer_answer(action: Mapping[str, Any]) -> int | None:
        value = action.get("answer")
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def _terminal(
        self,
        *,
        reason: str,
        reward: float,
        observation: dict[str, Any],
        components: dict[str, int],
        truncated: bool = False,
    ) -> StepResult:
        self._terminated = True
        self._last_reward = reward
        self._terminal_reason = reason
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=not truncated,
            truncated=truncated,
            info={"terminal_reason": reason, "reward_components": components},
        )

    def step(self, action: Mapping[str, Any]) -> StepResult:
        if self._terminated:
            raise RuntimeError("Episode already terminated; call reset before stepping again")
        if not isinstance(action, Mapping):
            raise TypeError("action must be a mapping")
        action_type = action.get("type")
        format_valid = int(action_type in {"tool_call", "final", "unavailable"})

        if action_type == "tool_call":
            name = action.get("name")
            arguments = action.get("arguments")
            record: dict[str, Any] = {
                "index": len(self._records),
                "type": "tool_call",
                "name": name,
                "arguments": dict(arguments) if isinstance(arguments, Mapping) else arguments,
                "valid": False,
                "semantic": None,
                "canonical_arguments": None,
                "result": None,
                "error": None,
            }
            tool = self._tool_by_name.get(name) if isinstance(name, str) else None
            if tool is None:
                record["error"] = "unknown_tool"
            elif not isinstance(arguments, Mapping):
                record["error"] = "arguments_not_object"
            else:
                try:
                    canonical, result = tool.execute(arguments)
                except (TypeError, ValueError) as exc:
                    record["error"] = f"schema_error:{exc}"
                else:
                    record.update(
                        {
                            "valid": True,
                            "semantic": tool.semantic,
                            "canonical_arguments": canonical,
                            "result": result,
                        }
                    )
            self._records.append(record)
            if len(self._records) >= self.max_steps:
                return self._terminal(
                    reason="step_budget_exhausted",
                    reward=0.0,
                    observation={"status": "truncated", "last_record": record},
                    components={"format": format_valid, "trace": 0, "answer": 0, "task": 0},
                    truncated=True,
                )
            return StepResult(
                observation={
                    "status": "tool_result" if record["valid"] else "tool_error",
                    "name": name,
                    "result": record["result"],
                    "error": record["error"],
                    "steps_remaining": self.max_steps - len(self._records),
                },
                reward=0.0,
                terminated=False,
                truncated=False,
                info={"reward_components": {"format": format_valid, "trace": 0, "answer": 0, "task": 0}},
            )

        if action_type == "final":
            answer = self._integer_answer(action)
            trace = int(self._gold_trace_matches())
            answer_correct = int(answer == self.task.expected_answer)
            available = int(self.view is not RegistryView.UNAVAILABLE)
            success = int(format_valid and trace and answer_correct and available)
            self._records.append(
                {
                    "index": len(self._records),
                    "type": "final",
                    "answer": answer,
                    "valid": bool(format_valid and answer is not None),
                }
            )
            return self._terminal(
                reason="success" if success else "incorrect_final",
                reward=float(success),
                observation={"status": "finished", "success": bool(success)},
                components={
                    "format": int(format_valid and answer is not None),
                    "trace": trace,
                    "answer": answer_correct,
                    "availability": available,
                    "task": success,
                },
            )

        if action_type == "unavailable":
            reason_text = action.get("reason")
            no_prior_calls = not any(record["type"] == "tool_call" for record in self._records)
            correct = int(
                self.view is RegistryView.UNAVAILABLE
                and isinstance(reason_text, str)
                and bool(reason_text.strip())
                and no_prior_calls
            )
            self._records.append(
                {
                    "index": len(self._records),
                    "type": "unavailable",
                    "reason": reason_text,
                    "valid": bool(format_valid and isinstance(reason_text, str) and reason_text.strip()),
                }
            )
            return self._terminal(
                reason="correct_unavailable" if correct else "incorrect_unavailable",
                reward=float(correct),
                observation={"status": "finished", "success": bool(correct)},
                components={"format": format_valid, "availability": correct, "task": correct},
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
            return self._terminal(
                reason="step_budget_exhausted",
                reward=0.0,
                observation={"status": "truncated", "error": "unknown_action_type"},
                components={"format": 0, "trace": 0, "answer": 0, "task": 0},
                truncated=True,
            )
        return StepResult(
            observation={"status": "action_error", "error": "unknown_action_type"},
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"reward_components": {"format": 0, "trace": 0, "answer": 0, "task": 0}},
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
        }
