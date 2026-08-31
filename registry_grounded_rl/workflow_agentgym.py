"""In-process AgentGym-RL client for the stateful workflow benchmark."""

from __future__ import annotations

from collections import defaultdict
import itertools
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .agentgym_compat import (
    AgentGymStepOutput,
    _single_query,
    balanced_capability_orbit_view,
    balanced_orbit_view,
    grouped_capability_orbit_view,
    selective_capability_orbit_view,
    stratified_capability_orbit_view,
    stratified_solvable_orbit_view,
)
from .environment import RegistryView
from .local_qwen import parse_agentgym_action
from .workflow_dataset import load_workflow_tasks
from .workflow_environment import RegistryWorkflowEnv


_WORKFLOW_CLIENT_SERIAL = itertools.count()
_GROUPED_ITEM_OCCURRENCES: defaultdict[tuple[int, int, str], int] = defaultdict(int)
_STRATIFIED_OCCURRENCES: defaultdict[tuple[int, str], int] = defaultdict(int)


def parse_workflow_address(address: str) -> tuple[Path, str, int]:
    parsed = urlparse(address)
    if parsed.scheme != "workflow":
        raise ValueError("Workflow benchmark requires workflow:///... task-file address")
    task_path = Path(unquote(parsed.path)).expanduser().resolve(strict=True)
    query = parse_qs(parsed.query, keep_blank_values=True)
    view = _single_query(query, "view", "original")
    schedules = {
        "orbit",
        "capability_orbit",
        "grouped_capability_orbit",
        "stratified_capability_orbit",
        "selective_capability_orbit",
        "stratified_solvable_orbit",
    }
    if view not in {item.value for item in RegistryView} | schedules:
        raise ValueError(f"Unsupported registry schedule: {view!r}")
    seed = int(_single_query(query, "seed", "1701"))
    return task_path, view, seed


class RegistryWorkflowAgentGymClient:
    conversation_start = (
        {
            "from": "human",
            "loss": None,
            "value": (
                "You are a stateful tool agent. Each task contains a live episode-local tool "
                "registry. Tool names, argument keys, and ordering can change, so act from "
                "descriptions and schemas. Discover internal IDs through read tools, persist all "
                "requested changes, and avoid preview-only distractors. Emit exactly one JSON "
                "object and no prose, using one of these forms:\n"
                '{"type":"tool_call","name":"...","arguments":{...}}\n'
                '{"type":"final","answer":"done"}\n'
                '{"type":"unavailable","reason":"..."}\n'
                "Reward is based on the exact terminal workspace state; changing unrelated "
                "tickets or claiming completion without persistence receives zero."
            ),
        },
        {
            "from": "gpt",
            "loss": False,
            "value": "Understood. I will follow the current registry and emit one action per turn.",
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
        path, schedule, seed = parse_workflow_address(env_server_base)
        query = parse_qs(urlparse(env_server_base).query, keep_blank_values=True)
        group_size = int(_single_query(query, "group_size", "6"))
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        self.tasks = load_workflow_tasks(path)
        self.data_len = len(self.tasks) if data_len <= 1 else data_len
        self.schedule = schedule
        self.seed = seed
        self.group_size = group_size
        self.client_serial = next(_WORKFLOW_CLIENT_SERIAL)
        self.env: RegistryWorkflowEnv | None = None
        self.info: dict[str, Any] | None = None

    def __len__(self) -> int:
        return self.data_len

    def _view_for(self, item_id: int) -> RegistryView:
        if self.schedule in {
            "stratified_capability_orbit",
            "selective_capability_orbit",
            "stratified_solvable_orbit",
        }:
            key = (self.seed, self.schedule)
            occurrence = _STRATIFIED_OCCURRENCES[key]
            _STRATIFIED_OCCURRENCES[key] += 1
            schedules = {
                "stratified_capability_orbit": stratified_capability_orbit_view,
                "selective_capability_orbit": selective_capability_orbit_view,
                "stratified_solvable_orbit": stratified_solvable_orbit_view,
            }
            schedule = schedules[self.schedule]
            return schedule(
                seed=self.seed,
                occurrence=occurrence,
                group_size=self.group_size,
            )
        if self.schedule == "grouped_capability_orbit":
            key = (self.seed, item_id, self.schedule)
            occurrence = _GROUPED_ITEM_OCCURRENCES[key]
            _GROUPED_ITEM_OCCURRENCES[key] += 1
            return grouped_capability_orbit_view(
                seed=self.seed,
                item_id=item_id,
                occurrence=occurrence,
                group_size=self.group_size,
            )
        if self.schedule == "capability_orbit":
            return balanced_capability_orbit_view(
                seed=self.seed,
                item_id=item_id,
                client_serial=self.client_serial,
                salt="workflow-capability",
            )
        if self.schedule != "orbit":
            return RegistryView(self.schedule)
        return balanced_orbit_view(
            seed=self.seed,
            item_id=item_id,
            client_serial=self.client_serial,
            salt="workflow",
        )

    def reset(self, idx: int) -> dict[str, Any]:
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise TypeError("item id must be an integer")
        if not 0 <= idx < len(self.tasks):
            raise IndexError(f"item id {idx} outside [0, {len(self.tasks)})")
        self.env = RegistryWorkflowEnv(self.tasks[idx], self._view_for(idx))
        self.info = self.env.reset()
        return dict(self.info)

    def observe(self) -> str:
        if self.info is None:
            raise RuntimeError("reset must be called before observe")
        return json.dumps(self.info, ensure_ascii=False, sort_keys=True)

    def step(self, action: str) -> AgentGymStepOutput:
        if self.env is None:
            raise RuntimeError("reset must be called before step")
        parsed = parse_agentgym_action(action)
        structured = parsed.action or {
            "type": "invalid_model_output",
            "raw": action,
            "parse_error": parsed.parse_error,
        }
        result = self.env.step(structured)
        state: dict[str, Any] = {
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
