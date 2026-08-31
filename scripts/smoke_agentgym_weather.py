#!/usr/bin/env python3
"""Run the official AgentGym Weather server, client, transition, and scorer.

This is an external-validity smoke test, not a reproducible training benchmark:
the official environment queries live Open-Meteo endpoints.  The script runs the
same task under the canonical and opaque registry surfaces and verifies that the
proxy preserves the official terminal reward.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import date, timedelta
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from registry_grounded_rl.stateful_agentgym import StatefulRegistryProxyClient  # noqa: E402


def _load_weather_client(agentenv_root: Path) -> type[Any]:
    """Load only WeatherEnvClient without importing AgentGym's training stack.

    The upstream controller package eagerly imports Transformers even though the
    Weather HTTP client only needs three tiny protocol types.  Lightweight stubs
    isolate that packaging dependency; the official ``weather.py`` source still
    provides all HTTP behavior exercised by this test.
    """

    agentenv_package = types.ModuleType("agentenv")
    agentenv_package.__path__ = [str(agentenv_root / "agentenv")]  # type: ignore[attr-defined]
    controller = types.ModuleType("agentenv.controller")
    controller_types = types.ModuleType("agentenv.controller.types")

    class BaseEnvClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    class BaseTask:
        pass

    class ConversationMessage(dict[str, Any]):
        pass

    @dataclass(frozen=True, slots=True)
    class StepOutput:
        state: str
        reward: float
        done: bool

    controller.BaseEnvClient = BaseEnvClient  # type: ignore[attr-defined]
    controller.BaseTask = BaseTask  # type: ignore[attr-defined]
    controller_types.ConversationMessage = ConversationMessage  # type: ignore[attr-defined]
    controller_types.StepOutput = StepOutput  # type: ignore[attr-defined]
    sys.modules["agentenv"] = agentenv_package
    sys.modules["agentenv.controller"] = controller
    sys.modules["agentenv.controller.types"] = controller_types

    path = agentenv_root / "agentenv" / "envs" / "weather.py"
    spec = importlib.util.spec_from_file_location("_registry_weather_client", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load AgentGym Weather client from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.WeatherEnvClient


def _wait_for_server(base_url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise RuntimeError(f"Weather server exited with {process.returncode}:\n{output}")
        try:
            response = requests.get(f"{base_url}/", timeout=1)
            if response.status_code == 200:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.2)
    raise TimeoutError(f"Weather server did not become ready: {last_error}")


def _action(proxy: StatefulRegistryProxyClient, canonical: str, arguments: dict[str, Any]) -> str:
    canonical_to_alias = {
        target: exposed for exposed, target in proxy.alias_to_canonical.items()
    }
    exposed = canonical_to_alias[canonical]
    return (
        "Thought: Execute the required tool according to its description.\n\n"
        f"Action: {exposed} with Action Input: {json.dumps(arguments, sort_keys=True)}"
    )


def _observation_payload(state: str) -> str:
    value = state.removeprefix("Observation: ")
    return value.rsplit("\nGive me one action.", 1)[0]


def _run_episode(
    weather_client_cls: type[Any], base_url: str, *, view: str, item_id: int
) -> dict[str, Any]:
    base = weather_client_cls(base_url, 343, timeout=30)
    proxy = StatefulRegistryProxyClient(base, schedule=view, seed=1701)
    proxy.reset(item_id)
    initial = proxy.observe()

    location_step = proxy.step(_action(proxy, "get_user_current_location", {}))
    location = _observation_payload(location_step.state)

    date_step = proxy.step(_action(proxy, "get_user_current_date", {}))
    current_date = date.fromisoformat(_observation_payload(date_step.state))
    yesterday = current_date - timedelta(days=1)

    geocode_step = proxy.step(
        _action(proxy, "get_latitude_longitude", {"name": location})
    )
    geocode = ast.literal_eval(_observation_payload(geocode_step.state))
    first = geocode["results"][0]

    weather_step = proxy.step(
        _action(
            proxy,
            "get_historical_temp",
            {
                "latitude": first["latitude"],
                "longitude": first["longitude"],
                "start_date": yesterday.isoformat(),
                "end_date": yesterday.isoformat(),
            },
        )
    )
    weather = ast.literal_eval(_observation_payload(weather_step.state))
    answer = weather["daily"]["temperature_2m_mean"][0]

    final_step = proxy.step(_action(proxy, "finish", {"answer": answer}))
    result = {
        "item_id": item_id,
        "view": view,
        "resolved_view": proxy.view.value if proxy.view is not None else None,
        "registry_present": "Registry view:" in initial,
        "location": location,
        "date": current_date.isoformat(),
        "answer_from_live_tool": answer,
        "terminal_reward": final_step.reward,
        "done": final_step.done,
        "canonical_trace": [
            row["canonical_name"] for row in proxy.records if row["type"] == "tool_call"
        ],
        "exposed_trace": [
            row["exposed_name"] for row in proxy.records if row["type"] == "tool_call"
        ],
    }
    proxy.close()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-root", type=Path, required=True)
    parser.add_argument("--agentenv-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=36101)
    parser.add_argument("--item-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tool_root = args.tool_root.expanduser().resolve(strict=True)
    agentenv_root = args.agentenv_root.expanduser().resolve(strict=True)
    base_url = f"http://127.0.0.1:{args.port}"
    server_env = os.environ.copy()
    server_env["PYTHONPATH"] = os.pathsep.join(
        [str(tool_root), str(tool_root / "Toolusage" / "toolusage")]
    )
    server_env["PROJECT_PATH"] = str(tool_root / "Toolusage")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "agentenv_weather.weather_server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
        ],
        cwd=tool_root,
        env=server_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_server(base_url, process, timeout=30)
        if str(agentenv_root) not in sys.path:
            sys.path.insert(0, str(agentenv_root))
        weather_client_cls = _load_weather_client(agentenv_root)
        episodes = [
            _run_episode(weather_client_cls, base_url, view=view, item_id=args.item_id)
            for view in ("original", "opaque_alias")
        ]
        dataset_path = tool_root / "Toolusage" / "data" / "weather.jsonl"
        with dataset_path.open("r", encoding="utf-8") as handle:
            frozen_row = next(
                json.loads(line)
                for index, line in enumerate(handle)
                if index == args.item_id
            )
        frozen_answer = frozen_row["additional_info"]["answer"]
        expected_trace = [
            "get_user_current_location",
            "get_user_current_date",
            "get_latitude_longitude",
            "get_historical_temp",
            "finish",
        ]
        compatibility_passed = bool(
            all(row["done"] and row["canonical_trace"] == expected_trace for row in episodes)
            and episodes[0]["answer_from_live_tool"] == episodes[1]["answer_from_live_tool"]
            and episodes[0]["exposed_trace"] != episodes[1]["exposed_trace"]
        )
        scorer_passed = all(row["terminal_reward"] == 1.0 for row in episodes)
        live_answer = episodes[0]["answer_from_live_tool"]
        result = {
            "purpose": "external-validity smoke test; live API, not training evidence",
            "official_environment": "AgentGym Toolusage Weather",
            "external_compatibility_passed": compatibility_passed,
            "official_frozen_scorer_passed": scorer_passed,
            "frozen_answer": frozen_answer,
            "live_answer": live_answer,
            "ground_truth_drift_detected": live_answer != frozen_answer,
            "usable_as_primary_training_benchmark": bool(scorer_passed),
            "exclusion_reason": (
                None
                if scorer_passed
                else "The live Open-Meteo answer no longer matches AgentGym's frozen target."
            ),
            "episodes": episodes,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if compatibility_passed else 1
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
