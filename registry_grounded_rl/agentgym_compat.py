"""In-process compatibility client for AgentGym-RL multi-turn rollouts.

AgentGym-RL asks an environment client to ``reset(item_id)``, ``observe()``,
and ``step(model_text)``.  This adapter deliberately stays in-process: the
environment is deterministic and has no heavyweight simulator that benefits
from a separate HTTP service.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .dataset import load_tasks
from .environment import RegistryGroundedEnv, RegistryView
from .evaluation import PRIMARY_VIEWS
from .local_qwen import parse_model_action

try:  # Use the official type when AgentGym is installed.
    from agentenv.controller.types import StepOutput as AgentGymStepOutput
except ImportError:  # Keep the project and unit tests dependency-light.

    @dataclass(frozen=True, slots=True)
    class AgentGymStepOutput:  # type: ignore[no-redef]
        state: str
        reward: float
        done: bool


_CLIENT_SERIAL = itertools.count()
_ORBIT_VIEWS = tuple(RegistryView(view) for view in PRIMARY_VIEWS)


def balanced_orbit_view(
    *, seed: int, item_id: int, client_serial: int, salt: str = "contract"
) -> RegistryView:
    """Round-robin views with a task-specific offset for balanced rollout groups."""

    key = f"{seed}:{item_id}:{salt}".encode()
    offset = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(_ORBIT_VIEWS)
    return _ORBIT_VIEWS[(offset + client_serial) % len(_ORBIT_VIEWS)]


def balanced_capability_orbit_view(
    *, seed: int, item_id: int, client_serial: int, salt: str = "workflow-capability"
) -> RegistryView:
    """Balance five solvable views plus the unavailable capability boundary."""

    views = (*_ORBIT_VIEWS, RegistryView.UNAVAILABLE)
    key = f"{seed}:{item_id}:{salt}".encode()
    offset = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(views)
    return views[(offset + client_serial) % len(views)]


def grouped_capability_orbit_view(
    *,
    seed: int,
    item_id: int,
    occurrence: int,
    group_size: int,
    salt: str = "workflow-grouped-capability",
) -> RegistryView:
    """Keep one registry state within a GRPO group and rotate it between groups."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return balanced_capability_orbit_view(
        seed=seed,
        item_id=item_id,
        client_serial=occurrence // group_size,
        salt=salt,
    )


def stratified_capability_orbit_view(
    *,
    seed: int,
    occurrence: int,
    group_size: int,
    salt: str = "workflow-stratified-capability",
) -> RegistryView:
    """Cycle capability states globally while holding each GRPO group fixed."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    views = (*_ORBIT_VIEWS, RegistryView.UNAVAILABLE)
    key = f"{seed}:{salt}".encode()
    offset = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(views)
    return views[(offset + occurrence // group_size) % len(views)]


def stratified_solvable_orbit_view(
    *,
    seed: int,
    occurrence: int,
    group_size: int,
    salt: str = "workflow-stratified-solvable",
) -> RegistryView:
    """Cycle solvable registry states globally while holding each GRPO group fixed."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    key = f"{seed}:{salt}".encode()
    offset = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(_ORBIT_VIEWS)
    return _ORBIT_VIEWS[(offset + occurrence // group_size) % len(_ORBIT_VIEWS)]


def selective_capability_orbit_view(
    *,
    seed: int,
    occurrence: int,
    group_size: int,
    salt: str = "workflow-selective-capability",
) -> RegistryView:
    """Alternate unavailable and solvable groups to match the balanced selective objective."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    group_index = occurrence // group_size
    if group_index % 2:
        return RegistryView.UNAVAILABLE
    key = f"{seed}:{salt}".encode()
    offset = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(_ORBIT_VIEWS)
    solvable_index = group_index // 2
    return _ORBIT_VIEWS[(offset + solvable_index) % len(_ORBIT_VIEWS)]


def _single_query(query: dict[str, list[str]], key: str, default: str) -> str:
    values = query.get(key, [default])
    if len(values) != 1:
        raise ValueError(f"Expected one {key!r} value")
    return values[0]


def parse_inproc_address(address: str) -> tuple[Path, str, int]:
    """Parse ``inproc:///tasks.jsonl?view=orbit&seed=1701``."""

    parsed = urlparse(address)
    if parsed.scheme != "inproc":
        raise ValueError("RegistryGrounded requires an inproc:///... task-file address")
    task_path = Path(unquote(parsed.path)).expanduser().resolve(strict=True)
    query = parse_qs(parsed.query, keep_blank_values=True)
    view = _single_query(query, "view", "original")
    if view not in {item.value for item in RegistryView} | {"orbit"}:
        raise ValueError(f"Unsupported registry schedule: {view!r}")
    seed = int(_single_query(query, "seed", "1701"))
    return task_path, view, seed


class RegistryGroundedAgentGymClient:
    """AgentGym client backed by :class:`RegistryGroundedEnv`."""

    conversation_start = (
        {
            "from": "human",
            "loss": None,
            "value": (
                "You are an executable tool agent. Each new task includes a user request and "
                "an episode-local tool registry. Follow tool descriptions and schemas, because "
                "names and argument aliases may change between episodes. Return exactly one "
                "action per turn using one of these forms:\n"
                '<tool_call>{"name":"...","arguments":{...}}</tool_call>\n'
                '<final>{"answer":INTEGER}</final>\n'
                '<unavailable>{"reason":"..."}</unavailable>\n'
                "Do not emit two action blocks in one turn. A correct answer without the exact "
                "executed semantic trace receives zero reward."
            ),
        },
        {
            "from": "gpt",
            "loss": False,
            "value": "Understood. I will emit exactly one executable action per turn.",
        },
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args: Any,
        timeout: int = 300,
        **kwargs: Any,
    ) -> None:
        del args, timeout, kwargs
        task_path, schedule, seed = parse_inproc_address(env_server_base)
        self.tasks = load_tasks(task_path)
        self.data_len = len(self.tasks) if data_len <= 1 else data_len
        self.schedule = schedule
        self.seed = seed
        self.client_serial = next(_CLIENT_SERIAL)
        self.env: RegistryGroundedEnv | None = None
        self.info: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.data_len

    def _view_for(self, item_id: int) -> RegistryView:
        if self.schedule != "orbit":
            return RegistryView(self.schedule)
        return balanced_orbit_view(
            seed=self.seed,
            item_id=item_id,
            client_serial=self.client_serial,
            salt="contract",
        )

    def reset(self, idx: int) -> dict[str, Any]:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError("item id must be an integer")
        if not 0 <= idx < len(self.tasks):
            raise IndexError(f"item id {idx} outside [0, {len(self.tasks)})")
        self.env = RegistryGroundedEnv(self.tasks[idx], self._view_for(idx))
        self.info = self.env.reset()
        return dict(self.info)

    def observe(self) -> str:
        if self.info is None:
            raise RuntimeError("reset must be called before observe")
        return json.dumps(self.info, ensure_ascii=False, sort_keys=True)

    def step(self, action: str) -> AgentGymStepOutput:
        if self.env is None:
            raise RuntimeError("reset must be called before step")
        parsed = parse_model_action(action)
        structured = parsed.action or {
            "type": "invalid_model_output",
            "raw": action,
            "parse_error": parsed.parse_error,
        }
        result = self.env.step(structured)
        state = {
            "observation": result.observation,
            "reward_components": result.info.get("reward_components", {}),
        }
        if parsed.parse_error:
            state["parse_error"] = parsed.parse_error
        self.info = state
        return AgentGymStepOutput(
            state=json.dumps(state, ensure_ascii=False, sort_keys=True),
            reward=result.reward,
            done=result.terminated or result.truncated,
        )

    def close(self) -> None:
        self.env = None
        self.info = None
