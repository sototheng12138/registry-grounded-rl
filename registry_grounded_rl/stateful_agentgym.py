"""Registry-surface proxy for stateful AgentGym Toolusage environments.

The proxy preserves the official downstream transition and scorer.  It changes
only the tool documentation shown to the model and translates an exposed action
name back to the canonical API before execution.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlparse, urlunparse

from .agentgym_compat import (
    AgentGymStepOutput,
    _CLIENT_SERIAL,
    _single_query,
    balanced_orbit_view,
)
from .environment import RegistryView


_ACTION_RE = re.compile(
    r"Action:\s*([A-Za-z0-9_.-]+)\s+with\s+Action\s+Input:\s*(\{.*?\})(?=\s*$)",
    re.DOTALL | re.IGNORECASE,
)
_TOOL_HEADER_RE = re.compile(r"(?m)^Name:\s*([A-Za-z0-9_]+)\(([^)]*)\)")


def parse_registry_http_address(address: str) -> tuple[str, str, int]:
    """Parse ``registry+http://host:port?view=orbit&seed=1701``."""

    parsed = urlparse(address)
    if parsed.scheme not in {"registry+http", "registry+https"}:
        raise ValueError("Expected registry+http:// or registry+https:// address")
    query = parse_qs(parsed.query, keep_blank_values=True)
    view = _single_query(query, "view", "original")
    if view not in {item.value for item in RegistryView} | {"orbit"}:
        raise ValueError(f"Unsupported registry schedule: {view!r}")
    seed = int(_single_query(query, "seed", "1701"))
    base_scheme = parsed.scheme.split("+", 1)[1]
    base_url = urlunparse((base_scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))
    return base_url, view, seed


class StatefulRegistryProxyClient:
    """Wrap one official Toolusage client without replacing its task scorer."""

    conversation_start = (
        {
            "from": "human",
            "loss": None,
            "value": (
                "You are an autonomous tool agent. The current task observation will include "
                "an episode-local registry. Follow its descriptions even when names or ordering "
                "change. Emit exactly one action in this form: Thought: ... followed by "
                "Action: TOOL with Action Input: JSON_OBJECT. Never invent an unlisted action."
            ),
        },
        {"from": "gpt", "loss": False, "value": "Understood."},
    )

    def __init__(self, base_client: Any, *, schedule: str, seed: int) -> None:
        self.base = base_client
        self.schedule = schedule
        self.seed = seed
        self.client_serial = next(_CLIENT_SERIAL)
        self.item_id: int | None = None
        self.view: RegistryView | None = None
        self.alias_to_canonical: dict[str, str] = {}
        self.distractors: set[str] = set()
        self.surface = ""
        self.records: list[dict[str, Any]] = []
        self.base_prompt = str(base_client.conversation_start[0]["value"])

    def __len__(self) -> int:
        return len(self.base)

    def _view_for(self, item_id: int) -> RegistryView:
        if self.schedule != "orbit":
            return RegistryView(self.schedule)
        return balanced_orbit_view(
            seed=self.seed,
            item_id=item_id,
            client_serial=self.client_serial,
            salt="stateful",
        )

    @staticmethod
    def _tool_names(prompt: str) -> tuple[str, ...]:
        names = tuple(match.group(1) for match in _TOOL_HEADER_RE.finditer(prompt))
        if not names:
            raise ValueError("No 'Name: function(...)' tool declarations found")
        return names

    def _opaque_name(self, canonical: str) -> str:
        assert self.item_id is not None
        digest = hashlib.sha256(f"{self.seed}:{self.item_id}:{canonical}".encode()).hexdigest()
        return f"unit_{digest[:8]}"

    @staticmethod
    def _replace_identifiers(text: str, mapping: dict[str, str]) -> str:
        for source in sorted(mapping, key=len, reverse=True):
            text = re.sub(rf"\b{re.escape(source)}\b", mapping[source], text)
        return text

    @staticmethod
    def _reorder_blocks(prompt: str, *, reverse: bool) -> str:
        matches = list(_TOOL_HEADER_RE.finditer(prompt))
        if len(matches) < 2:
            return prompt
        tail_markers = [
            prompt.find("\n\nIf you want", matches[-1].start()),
            prompt.find("\n\nYour response must", matches[-1].start()),
        ]
        tail_candidates = [index for index in tail_markers if index >= 0]
        tail_start = min(tail_candidates) if tail_candidates else len(prompt)
        prefix = prompt[: matches[0].start()]
        blocks: list[str] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else tail_start
            blocks.append(prompt[match.start() : end])
        if reverse:
            blocks.reverse()
        return prefix + "".join(blocks) + prompt[tail_start:]

    def _build_surface(self) -> None:
        assert self.view is not None
        names = self._tool_names(self.base_prompt)
        self.alias_to_canonical = {name: name for name in names}
        self.distractors.clear()
        surface = self.base_prompt
        if self.view is RegistryView.ORDER:
            surface = self._reorder_blocks(surface, reverse=True)
        elif self.view is RegistryView.SCHEMA_SURFACE:
            surface = surface.replace("Name:", "Function:")
            surface = surface.replace("Parameters:", "Arguments:")
            surface = surface.replace("Returns:", "Output:")
        elif self.view is RegistryView.OPAQUE_ALIAS:
            canonical_to_alias = {name: self._opaque_name(name) for name in names}
            self.alias_to_canonical = {
                alias: canonical for canonical, alias in canonical_to_alias.items()
            }
            surface = self._replace_identifiers(surface, canonical_to_alias)
        elif self.view is RegistryView.HARD_DISTRACTOR:
            distractors = {f"{name}_preview" for name in names[: min(3, len(names))]}
            self.distractors = distractors
            additions = "\n\n".join(
                f"Name: {name}()\nDescription: Preview-only analogue; it does not execute the action."
                for name in sorted(distractors)
            )
            surface = surface + "\n\nPotentially similar utilities:\n" + additions
        elif self.view is RegistryView.UNAVAILABLE:
            raise ValueError("Unavailable is not used for stateful external-validity training")
        self.surface = surface

    def reset(self, idx: int) -> Any:
        self.item_id = idx
        self.view = self._view_for(idx)
        response = self.base.reset(idx)
        self.records.clear()
        self._build_surface()
        return response

    def observe(self) -> str:
        if self.view is None:
            raise RuntimeError("reset must be called before observe")
        observation = self.base.observe()
        return (
            f"Registry view: {self.view.value}\n\n{self.surface}\n\n"
            f"Current task observation:\n{observation}"
        )

    def step(self, action: str) -> AgentGymStepOutput:
        if self.view is None:
            raise RuntimeError("reset must be called before step")
        matches = _ACTION_RE.findall(action)
        if len(matches) != 1:
            self.records.append({"type": "malformed", "raw": action})
            return AgentGymStepOutput(
                state="Format error: emit exactly one Action with one JSON Action Input.",
                reward=0.0,
                done=False,
            )
        exposed_name, raw_arguments = matches[0]
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            self.records.append({"type": "malformed", "name": exposed_name, "raw": action})
            return AgentGymStepOutput(state="Format error: Action Input must be JSON.", reward=0.0, done=False)
        if not isinstance(arguments, dict):
            self.records.append({"type": "malformed", "name": exposed_name, "raw": action})
            return AgentGymStepOutput(state="Format error: Action Input must be an object.", reward=0.0, done=False)
        if exposed_name in self.distractors:
            self.records.append({"type": "distractor", "name": exposed_name, "arguments": arguments})
            return AgentGymStepOutput(
                state="Tool error: this preview utility cannot execute the requested action.",
                reward=0.0,
                done=False,
            )
        canonical = self.alias_to_canonical.get(exposed_name)
        if canonical is None:
            self.records.append({"type": "unknown", "name": exposed_name, "arguments": arguments})
            return AgentGymStepOutput(state="Tool error: unknown action name.", reward=0.0, done=False)
        translated = (
            f"Thought: Execute the selected registry action.\n\n"
            f"Action: {canonical} with Action Input: {json.dumps(arguments, sort_keys=True)}"
        )
        output = self.base.step(translated)
        self.records.append(
            {
                "type": "tool_call",
                "exposed_name": exposed_name,
                "canonical_name": canonical,
                "arguments": arguments,
                "reward": output.reward,
                "done": output.done,
            }
        )
        return AgentGymStepOutput(state=output.state, reward=output.reward, done=output.done)

    def close(self) -> Any:
        close = getattr(self.base, "close", None)
        return close() if close is not None else None


class RegistryTodoAgentGymClient(StatefulRegistryProxyClient):
    def __init__(self, env_server_base: str, data_len: int, *args: Any, timeout: int = 300, **kwargs: Any):
        from agentenv.envs import TodoEnvClient

        base_url, schedule, seed = parse_registry_http_address(env_server_base)
        base = TodoEnvClient(base_url, data_len, *args, timeout=timeout, **kwargs)
        super().__init__(base, schedule=schedule, seed=seed)


class RegistryWeatherAgentGymClient(StatefulRegistryProxyClient):
    def __init__(self, env_server_base: str, data_len: int, *args: Any, timeout: int = 300, **kwargs: Any):
        from agentenv.envs import WeatherEnvClient

        base_url, schedule, seed = parse_registry_http_address(env_server_base)
        base = WeatherEnvClient(base_url, data_len, *args, timeout=timeout, **kwargs)
        super().__init__(base, schedule=schedule, seed=seed)
