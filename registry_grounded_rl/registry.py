"""Episode-local tool registries and lossless argument translation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Any, Mapping

from .tasks import execute_operation


DESCRIPTIONS = {
    "add": "Return the exact sum of two integers.",
    "subtract": "Return the first integer minus the second integer.",
    "multiply": "Return the exact product of two integers.",
    "maximum": "Return the larger of two integers.",
    "add_then_one": "Add two integers and then increase the sum by one.",
    "multiply_then_one": "Multiply two integers and then increase the product by one.",
}
CANONICAL_NAMES = {
    "add": "add_integers",
    "subtract": "subtract_integers",
    "multiply": "multiply_integers",
    "maximum": "maximum_integer",
    "add_then_one": "add_integers_plus_one",
    "multiply_then_one": "multiply_integers_plus_one",
}


@dataclass(frozen=True, slots=True)
class ToolSpec:
    semantic: str
    exposed_name: str
    public_lhs: str = "lhs"
    public_rhs: str = "rhs"
    reverse_schema_order: bool = False

    @property
    def public_arguments(self) -> tuple[str, str]:
        return (self.public_lhs, self.public_rhs)

    def schema(self) -> dict[str, Any]:
        first, second = self.public_arguments
        ordered = (second, first) if self.reverse_schema_order else (first, second)
        properties = {
            name: {
                "type": "integer",
                "description": "first integer" if name == first else "second integer",
            }
            for name in ordered
        }
        return {
            "type": "function",
            "function": {
                "name": self.exposed_name,
                "description": DESCRIPTIONS[self.semantic],
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [first, second],
                    "properties": properties,
                },
            },
        }

    def translate(self, arguments: Mapping[str, Any]) -> dict[str, int]:
        if set(arguments) != set(self.public_arguments):
            raise ValueError(
                f"Expected exactly arguments {sorted(self.public_arguments)}, "
                f"received {sorted(arguments)}"
            )
        lhs = arguments[self.public_lhs]
        rhs = arguments[self.public_rhs]
        if isinstance(lhs, bool) or not isinstance(lhs, int):
            raise ValueError(f"{self.public_lhs} must be an integer")
        if isinstance(rhs, bool) or not isinstance(rhs, int):
            raise ValueError(f"{self.public_rhs} must be an integer")
        return {"lhs": lhs, "rhs": rhs}

    def execute(self, arguments: Mapping[str, Any]) -> tuple[dict[str, int], int]:
        canonical = self.translate(arguments)
        return canonical, execute_operation(self.semantic, canonical["lhs"], canonical["rhs"])


def _opaque_name(task_id: str, semantic: str) -> str:
    digest = hashlib.sha256(f"{task_id}:{semantic}".encode("utf-8")).hexdigest()[:8]
    return f"unit_{digest}"


def build_registry(task_id: str, view: str, missing_semantic: str | None = None) -> tuple[ToolSpec, ...]:
    """Build a registry whose semantic layer stays hidden from model-facing schemas."""

    if view not in {
        "original",
        "order",
        "schema_surface",
        "opaque_alias",
        "hard_distractor",
        "unavailable",
    }:
        raise ValueError(f"Unsupported registry view: {view!r}")

    semantics = list(("add", "subtract", "multiply", "maximum"))
    if view == "unavailable":
        if missing_semantic not in semantics:
            raise ValueError("unavailable view requires a primary missing_semantic")
        semantics.remove(str(missing_semantic))
    if view == "hard_distractor":
        semantics.extend(("add_then_one", "multiply_then_one"))

    tools: list[ToolSpec] = []
    for semantic in semantics:
        if view == "opaque_alias":
            tools.append(
                ToolSpec(
                    semantic=semantic,
                    exposed_name=_opaque_name(task_id, semantic),
                    public_lhs="input_a",
                    public_rhs="input_b",
                )
            )
        else:
            tools.append(
                ToolSpec(
                    semantic=semantic,
                    exposed_name=CANONICAL_NAMES[semantic],
                    reverse_schema_order=view == "schema_surface",
                )
            )
    if view in {"order", "hard_distractor"}:
        rng = random.Random(int(hashlib.sha256(f"{task_id}:{view}".encode()).hexdigest()[:16], 16))
        rng.shuffle(tools)
    return tuple(tools)

