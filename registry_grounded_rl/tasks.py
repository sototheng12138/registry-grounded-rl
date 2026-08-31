"""Deterministic procedural tasks with executable, auditable ground truth."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Iterable, Mapping


PRIMARY_OPERATIONS = ("add", "subtract", "multiply", "maximum")
PREVIOUS = "$previous"


def execute_operation(name: str, lhs: int, rhs: int) -> int:
    """Execute one semantic operation, including non-gold distractor tools."""

    if name == "add":
        return lhs + rhs
    if name == "subtract":
        return lhs - rhs
    if name == "multiply":
        return lhs * rhs
    if name == "maximum":
        return max(lhs, rhs)
    if name == "add_then_one":
        return lhs + rhs + 1
    if name == "multiply_then_one":
        return lhs * rhs + 1
    raise ValueError(f"Unsupported semantic operation: {name!r}")


@dataclass(frozen=True, slots=True)
class OperationStep:
    semantic: str
    lhs: int | str
    rhs: int

    def __post_init__(self) -> None:
        if self.semantic not in PRIMARY_OPERATIONS:
            raise ValueError(f"Unsupported gold operation: {self.semantic!r}")
        if self.lhs != PREVIOUS and not isinstance(self.lhs, int):
            raise ValueError("lhs must be an integer or $previous")
        if not isinstance(self.rhs, int):
            raise ValueError("rhs must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {"semantic": self.semantic, "lhs": self.lhs, "rhs": self.rhs}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperationStep":
        lhs = value["lhs"]
        return cls(
            semantic=str(value["semantic"]),
            lhs=int(lhs) if lhs != PREVIOUS else PREVIOUS,
            rhs=int(value["rhs"]),
        )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    split: str
    request: str
    steps: tuple[OperationStep, ...]
    template_family: int

    def __post_init__(self) -> None:
        if not self.task_id or not self.request.strip():
            raise ValueError("task_id and request must be non-empty")
        if self.split not in {"train", "dev", "test", "smoke"}:
            raise ValueError(f"Unsupported split: {self.split!r}")
        if not 1 <= len(self.steps) <= 3:
            raise ValueError("Tasks must contain one to three operations")
        if self.steps[0].lhs == PREVIOUS:
            raise ValueError("The first operation cannot consume $previous")
        if any(step.lhs != PREVIOUS for step in self.steps[1:]):
            raise ValueError("Later operations must consume $previous")

    @property
    def expected_answer(self) -> int:
        previous: int | None = None
        for step in self.steps:
            lhs = previous if step.lhs == PREVIOUS else int(step.lhs)
            if lhs is None:
                raise RuntimeError("Invalid task: previous result is unavailable")
            previous = execute_operation(step.semantic, lhs, step.rhs)
        assert previous is not None
        return previous

    def gold_calls(self) -> tuple[dict[str, Any], ...]:
        calls: list[dict[str, Any]] = []
        previous: int | None = None
        for step in self.steps:
            lhs = previous if step.lhs == PREVIOUS else int(step.lhs)
            if lhs is None:
                raise RuntimeError("Invalid task: previous result is unavailable")
            result = execute_operation(step.semantic, lhs, step.rhs)
            calls.append(
                {
                    "semantic": step.semantic,
                    "arguments": {"lhs": lhs, "rhs": step.rhs},
                    "result": result,
                }
            )
            previous = result
        return tuple(calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "request": self.request,
            "steps": [step.to_dict() for step in self.steps],
            "template_family": self.template_family,
            "expected_answer": self.expected_answer,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        task = cls(
            task_id=str(value["task_id"]),
            split=str(value["split"]),
            request=str(value["request"]),
            steps=tuple(OperationStep.from_dict(item) for item in value["steps"]),
            template_family=int(value["template_family"]),
        )
        if "expected_answer" in value and int(value["expected_answer"]) != task.expected_answer:
            raise ValueError(f"Stored expected_answer disagrees for {task.task_id}")
        return task


_VERBS = {
    "add": ("add", "sum"),
    "subtract": ("subtract", "take away"),
    "multiply": ("multiply", "take the product of"),
    "maximum": ("take the larger of", "compute the maximum of"),
}


def _render_request(steps: tuple[OperationStep, ...], family: int) -> str:
    fragments: list[str] = []
    for index, step in enumerate(steps):
        verb = _VERBS[step.semantic][family % 2]
        lhs = str(step.lhs) if step.lhs != PREVIOUS else "the previous result"
        if family in {0, 2}:
            fragment = f"{verb} {lhs} and {step.rhs}"
        else:
            fragment = f"{verb} the values {lhs} and {step.rhs}"
        fragments.append(fragment)
    if family == 0:
        body = ", then ".join(fragments)
        return f"Use the available tools to {body}. Return the final integer."
    if family == 1:
        numbered = "; ".join(f"step {index + 1}: {part}" for index, part in enumerate(fragments))
        return f"Complete this calculation with tools — {numbered}. Report the final integer."
    if family == 2:
        body = ". Next, ".join(part.capitalize() for part in fragments)
        return f"Perform the requested tool workflow. {body}. Give the resulting integer."
    body = " followed by ".join(fragments)
    return f"Execute, rather than estimate, the following workflow: {body}. Output its final integer."


def generate_tasks(
    count: int,
    *,
    seed: int,
    split: str,
    template_families: Iterable[int] | None = None,
) -> tuple[TaskSpec, ...]:
    """Generate deterministic tasks without reading model or evaluation outputs."""

    if count <= 0:
        raise ValueError("count must be positive")
    families = tuple(template_families if template_families is not None else range(4))
    if not families:
        raise ValueError("At least one template family is required")
    rng = random.Random(seed)
    tasks: list[TaskSpec] = []
    for index in range(count):
        horizon = 1 + rng.randrange(3)
        family = families[index % len(families)]
        steps: list[OperationStep] = []
        for step_index in range(horizon):
            semantic = PRIMARY_OPERATIONS[rng.randrange(len(PRIMARY_OPERATIONS))]
            lhs: int | str = rng.randint(2, 20) if step_index == 0 else PREVIOUS
            rhs = rng.randint(2, 12)
            steps.append(OperationStep(semantic=semantic, lhs=lhs, rhs=rhs))
        frozen_steps = tuple(steps)
        tasks.append(
            TaskSpec(
                task_id=f"rg-{split}-{seed}-{index:05d}",
                split=split,
                request=_render_request(frozen_steps, family),
                steps=frozen_steps,
                template_family=family,
            )
        )
    return tuple(tasks)

