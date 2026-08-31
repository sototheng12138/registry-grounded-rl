from dataclasses import dataclass

from registry_grounded_rl.stateful_agentgym import (
    StatefulRegistryProxyClient,
    parse_registry_http_address,
)


@dataclass
class FakeStep:
    state: str
    reward: float
    done: bool


class FakeToolClient:
    conversation_start = (
        {
            "value": (
                "Use the tools below.\n"
                "Name: get_items()\nDescription: Read items.\n\n"
                "Name: finish(answer)\nDescription: Finish the task.\n\n"
                "Your response must contain one action."
            )
        },
        {"value": "OK"},
    )

    def __init__(self) -> None:
        self.last_action = ""

    def __len__(self) -> int:
        return 4

    def reset(self, idx: int) -> dict[str, int]:
        return {"id": idx}

    def observe(self) -> str:
        return "Find all items."

    def step(self, action: str) -> FakeStep:
        self.last_action = action
        return FakeStep(state="done", reward=float("Action: finish" in action), done=True)


def test_registry_http_address_strips_proxy_configuration() -> None:
    base, view, seed = parse_registry_http_address(
        "registry+http://127.0.0.1:9000?view=orbit&seed=9"
    )
    assert base == "http://127.0.0.1:9000"
    assert view == "orbit"
    assert seed == 9


def test_opaque_proxy_translates_only_surface_name() -> None:
    base = FakeToolClient()
    proxy = StatefulRegistryProxyClient(base, schedule="opaque_alias", seed=3)
    proxy.reset(0)
    observation = proxy.observe()
    alias = next(name for name, canonical in proxy.alias_to_canonical.items() if canonical == "finish")
    assert alias in observation
    assert "Name: finish(" not in observation
    result = proxy.step(
        f'Thought: done.\n\nAction: {alias} with Action Input: {{"answer":"done"}}'
    )
    assert result.reward == 1.0
    assert "Action: finish" in base.last_action
    assert proxy.records[-1]["exposed_name"] == alias


def test_distractor_is_rejected_without_downstream_execution() -> None:
    base = FakeToolClient()
    proxy = StatefulRegistryProxyClient(base, schedule="hard_distractor", seed=3)
    proxy.reset(0)
    distractor = sorted(proxy.distractors)[0]
    result = proxy.step(f"Action: {distractor} with Action Input: {{}}")
    assert result.done is False
    assert result.reward == 0.0
    assert base.last_action == ""

