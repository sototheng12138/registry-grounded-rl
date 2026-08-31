"""Reproducible analysis helpers for matched GRPO pilots and frozen evaluation."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping, Sequence

from .local_qwen import parse_agentgym_action
from .environment import RegistryView
from .workflow_environment import RegistryWorkflowEnv
from .workflow_tasks import WorkflowTaskSpec


PRIMARY_VIEWS = (
    "original",
    "order",
    "schema_surface",
    "opaque_alias",
    "hard_distractor",
)
ALL_VIEWS = (*PRIMARY_VIEWS, "unavailable")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\s+-")
METRIC_RE = re.compile(r"(?:^|\s-\s)([A-Za-z0-9_./-]+):([-+0-9.eE]+)")


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float]:
    """Return the two-sided Wilson interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - half_width, center + half_width]


def exact_mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, float | int]:
    """Exact two-sided McNemar test using only discordant paired outcomes."""

    if len(left) != len(right):
        raise ValueError("paired outcomes must have equal length")
    improved = sum(not before and after for before, after in zip(left, right, strict=True))
    regressed = sum(before and not after for before, after in zip(left, right, strict=True))
    discordant = improved + regressed
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
        p_value = min(1.0, 2 * tail / (2**discordant))
    return {
        "improved": improved,
        "regressed": regressed,
        "discordant": discordant,
        "p_value_two_sided": p_value,
    }


def paired_task_bootstrap(
    left: Mapping[str, Mapping[int, bool]],
    right: Mapping[str, Mapping[int, bool]],
    *,
    views: Sequence[str] = PRIMARY_VIEWS,
    samples: int = 10_000,
    seed: int = 1701,
) -> dict[str, float | int | list[float]]:
    """Bootstrap a paired rate difference while preserving within-task view correlation."""

    task_ids = sorted(set.intersection(*(set(left[view]) & set(right[view]) for view in views)))
    if not task_ids:
        raise ValueError("no paired tasks")
    per_task = [
        sum(float(right[view][item_id]) - float(left[view][item_id]) for view in views) / len(views)
        for item_id in task_ids
    ]
    observed = sum(per_task) / len(per_task)
    generator = random.Random(seed)
    draws = sorted(
        sum(generator.choice(per_task) for _ in task_ids) / len(task_ids) for _ in range(samples)
    )

    def percentile(probability: float) -> float:
        position = probability * (len(draws) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return draws[lower]
        weight = position - lower
        return draws[lower] * (1 - weight) + draws[upper] * weight

    return {
        "unit": "task_with_view_correlation_preserved",
        "tasks": len(task_ids),
        "views_per_task": len(views),
        "samples": samples,
        "seed": seed,
        "difference": observed,
        "ci95_percentile": [percentile(0.025), percentile(0.975)],
        "bootstrap_probability_positive": sum(value > 0 for value in draws) / len(draws),
    }


def paired_weighted_task_bootstrap(
    left: Mapping[str, Mapping[int, bool]],
    right: Mapping[str, Mapping[int, bool]],
    *,
    view_weights: Mapping[str, float],
    samples: int = 10_000,
    seed: int = 1701,
) -> dict[str, float | int | list[float] | dict[str, float]]:
    """Bootstrap a paired weighted score while keeping all views of a task together."""

    if not view_weights:
        raise ValueError("view_weights must not be empty")
    if any(weight < 0 for weight in view_weights.values()):
        raise ValueError("view weights must be non-negative")
    weight_sum = sum(view_weights.values())
    if weight_sum <= 0:
        raise ValueError("view weights must have a positive sum")
    normalized = {view: weight / weight_sum for view, weight in view_weights.items()}
    task_ids = sorted(
        set.intersection(*(set(left[view]) & set(right[view]) for view in normalized))
    )
    if not task_ids:
        raise ValueError("no paired tasks")
    per_task = [
        sum(
            normalized[view]
            * (float(right[view][item_id]) - float(left[view][item_id]))
            for view in normalized
        )
        for item_id in task_ids
    ]
    observed = sum(per_task) / len(per_task)
    generator = random.Random(seed)
    draws = sorted(
        sum(generator.choice(per_task) for _ in task_ids) / len(task_ids) for _ in range(samples)
    )

    def percentile(probability: float) -> float:
        position = probability * (len(draws) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return draws[lower]
        weight = position - lower
        return draws[lower] * (1 - weight) + draws[upper] * weight

    return {
        "unit": "task_with_weighted_views_preserved",
        "tasks": len(task_ids),
        "view_weights": normalized,
        "samples": samples,
        "seed": seed,
        "difference": observed,
        "ci95_percentile": [percentile(0.025), percentile(0.975)],
        "bootstrap_probability_positive": sum(value > 0 for value in draws) / len(draws),
    }


def selective_harmonic_score(solvable_rate: float, unavailable_rate: float) -> float:
    """Harmonic mean that penalizes collapse on either execution or correct stopping."""

    if not 0 <= solvable_rate <= 1 or not 0 <= unavailable_rate <= 1:
        raise ValueError("rates must lie in [0, 1]")
    denominator = solvable_rate + unavailable_rate
    return 0.0 if denominator == 0 else 2 * solvable_rate * unavailable_rate / denominator


def paired_selective_harmonic_bootstrap(
    left: Mapping[str, Mapping[int, bool]],
    right: Mapping[str, Mapping[int, bool]],
    *,
    solvable_views: Sequence[str] = PRIMARY_VIEWS,
    unavailable_view: str = "unavailable",
    samples: int = 10_000,
    seed: int = 1701,
) -> dict[str, float | int | list[float]]:
    """Bootstrap the paired difference in execution/stopping harmonic mean by task."""

    views = (*solvable_views, unavailable_view)
    task_ids = sorted(set.intersection(*(set(left[view]) & set(right[view]) for view in views)))
    if not task_ids:
        raise ValueError("no paired tasks")

    def score(outcomes: Mapping[str, Mapping[int, bool]], selected: Sequence[int]) -> float:
        solvable = sum(
            float(outcomes[view][item_id]) for item_id in selected for view in solvable_views
        ) / (len(selected) * len(solvable_views))
        unavailable = sum(
            float(outcomes[unavailable_view][item_id]) for item_id in selected
        ) / len(selected)
        return selective_harmonic_score(solvable, unavailable)

    observed_left = score(left, task_ids)
    observed_right = score(right, task_ids)
    generator = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [generator.choice(task_ids) for _ in task_ids]
        draws.append(score(right, selected) - score(left, selected))
    draws.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(draws) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return draws[lower]
        weight = position - lower
        return draws[lower] * (1 - weight) + draws[upper] * weight

    return {
        "unit": "task_with_all_selective_views_preserved",
        "tasks": len(task_ids),
        "solvable_views_per_task": len(solvable_views),
        "samples": samples,
        "seed": seed,
        "left_score": observed_left,
        "right_score": observed_right,
        "difference": observed_right - observed_left,
        "ci95_percentile": [percentile(0.025), percentile(0.975)],
        "bootstrap_probability_positive": sum(value > 0 for value in draws) / len(draws),
    }


def hierarchical_paired_bootstrap(
    left_runs: Sequence[Mapping[str, Mapping[int, bool]]],
    right_runs: Sequence[Mapping[str, Mapping[int, bool]]],
    *,
    solvable_views: Sequence[str] = PRIMARY_VIEWS,
    unavailable_view: str = "unavailable",
    samples: int = 10_000,
    seed: int = 2701,
) -> dict[str, Any]:
    """Resample training seeds, then paired tasks, without treating tasks as seed replicates."""

    if not left_runs or len(left_runs) != len(right_runs):
        raise ValueError("left and right must contain the same positive number of runs")
    views = (*solvable_views, unavailable_view)
    task_ids_by_run = [
        sorted(
            set.intersection(
                *(set(left[view]) & set(right[view]) for view in views)
            )
        )
        for left, right in zip(left_runs, right_runs, strict=True)
    ]
    if any(not task_ids for task_ids in task_ids_by_run):
        raise ValueError("every run pair must contain paired tasks")

    def metrics(
        left: Mapping[str, Mapping[int, bool]],
        right: Mapping[str, Mapping[int, bool]],
        selected: Sequence[int],
    ) -> dict[str, float]:
        denominator = len(selected) * len(solvable_views)
        left_solvable = sum(
            float(left[view][item_id]) for item_id in selected for view in solvable_views
        ) / denominator
        right_solvable = sum(
            float(right[view][item_id]) for item_id in selected for view in solvable_views
        ) / denominator
        left_unavailable = sum(
            float(left[unavailable_view][item_id]) for item_id in selected
        ) / len(selected)
        right_unavailable = sum(
            float(right[unavailable_view][item_id]) for item_id in selected
        ) / len(selected)
        return {
            "primary_difference": right_solvable - left_solvable,
            "unavailable_difference": right_unavailable - left_unavailable,
            "selective_harmonic_difference": selective_harmonic_score(
                right_solvable, right_unavailable
            )
            - selective_harmonic_score(left_solvable, left_unavailable),
        }

    per_seed = [
        metrics(left, right, task_ids)
        for left, right, task_ids in zip(
            left_runs, right_runs, task_ids_by_run, strict=True
        )
    ]
    observed = {
        key: sum(item[key] for item in per_seed) / len(per_seed) for key in per_seed[0]
    }
    generator = random.Random(seed)
    draws: dict[str, list[float]] = {key: [] for key in observed}
    for _ in range(samples):
        seed_indices = [generator.randrange(len(left_runs)) for _ in left_runs]
        sampled_metrics = []
        for run_index in seed_indices:
            task_ids = task_ids_by_run[run_index]
            selected = [generator.choice(task_ids) for _ in task_ids]
            sampled_metrics.append(
                metrics(left_runs[run_index], right_runs[run_index], selected)
            )
        for key in draws:
            draws[key].append(
                sum(item[key] for item in sampled_metrics) / len(sampled_metrics)
            )

    def percentile(values: Sequence[float], probability: float) -> float:
        position = probability * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    results: dict[str, Any] = {}
    for key, values in draws.items():
        values.sort()
        results[key] = {
            "difference": observed[key],
            "ci95_percentile": [percentile(values, 0.025), percentile(values, 0.975)],
            "bootstrap_probability_positive": sum(value > 0 for value in values) / len(values),
        }
    return {
        "unit": "training_seed_then_paired_task",
        "training_seeds": len(left_runs),
        "tasks_per_seed": [len(task_ids) for task_ids in task_ids_by_run],
        "samples": samples,
        "seed": seed,
        "per_seed": per_seed,
        "metrics": results,
    }


def parse_training_metrics(path: Path) -> list[dict[str, float | int]]:
    """Extract one stable metric record per optimizer update from a VERL log."""

    text = ANSI_RE.sub("", path.read_text(errors="replace"))
    records: list[dict[str, float | int]] = []
    for line in text.splitlines():
        if "critic/score/mean" not in line:
            continue
        step_match = STEP_RE.search(line)
        if step_match is None:
            continue
        record: dict[str, float | int] = {"step": int(step_match.group(1))}
        for key, value in METRIC_RE.findall(line):
            record[key] = float(value)
        records.append(record)
    by_step = {int(record["step"]): record for record in records}
    return [by_step[step] for step in sorted(by_step)]


def load_trajectory_rows(run_dir: Path) -> list[dict[str, Any]]:
    files = sorted((run_dir / "trajectories").rglob("*.json"))
    rows: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"trajectory payload is not a list: {path}")
        rows.extend(payload)
    return rows


def generated_trajectory_fingerprint(row: Mapping[str, Any]) -> str:
    """Return a stable fingerprint of model messages after the initial environment state."""

    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("episode conversations must be a list")
    initial_state_index: int | None = None
    for index, message in enumerate(conversations):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            candidate = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("environment_schema") == (
            "registry-grounded-rl/stateful-workflow-env-v2"
        ):
            initial_state_index = index
            break
    if initial_state_index is None:
        raise ValueError("workflow initial state is absent from trajectory")
    generated = [
        {"role": message.get("role"), "content": message.get("content")}
        for message in conversations[initial_state_index + 1 :]
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    return json.dumps(generated, ensure_ascii=False, separators=(",", ":"))


def summarize_rollout_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    rollout_n: int,
    tasks: Mapping[int, WorkflowTaskSpec] | None = None,
) -> dict[str, Any]:
    """Audit candidate diversity and reward variance in contiguous GRPO groups."""

    if rollout_n <= 1:
        raise ValueError("rollout_n must be greater than one")
    if not rows or len(rows) % rollout_n:
        raise ValueError("rows must contain complete non-empty rollout groups")
    groups = []
    for offset in range(0, len(rows), rollout_n):
        group = rows[offset : offset + rollout_n]
        item_ids = {int(row["item_id"]) for row in group}
        if len(item_ids) != 1:
            raise ValueError(f"rollout group mixes item IDs: {sorted(item_ids)}")
        item_id = next(iter(item_ids))
        rewards = [float(row["reward"]) for row in group]
        unique_trajectories = len({generated_trajectory_fingerprint(row) for row in group})
        view = None
        if tasks is not None:
            if item_id not in tasks:
                raise ValueError(f"missing task specification for item {item_id}")
            inferred = {infer_workflow_view(row, tasks[item_id]) for row in group}
            if len(inferred) != 1:
                raise ValueError(f"rollout group mixes registry views: {sorted(inferred)}")
            view = next(iter(inferred))
        groups.append(
            {
                "item_id": item_id,
                "view": view,
                "unique_trajectories": unique_trajectories,
                "successes": sum(reward == 1.0 for reward in rewards),
                "mixed_reward": min(rewards) != max(rewards),
            }
        )
    unavailable = [group for group in groups if group["view"] == "unavailable"]
    return {
        "episodes": len(rows),
        "rollout_n": rollout_n,
        "groups": len(groups),
        "mean_unique_trajectories_per_group": sum(
            group["unique_trajectories"] for group in groups
        )
        / len(groups),
        "fully_duplicated_groups": sum(group["unique_trajectories"] == 1 for group in groups),
        "mixed_reward_groups": sum(group["mixed_reward"] for group in groups),
        "zero_reward_variance_groups": sum(not group["mixed_reward"] for group in groups),
        "successes": sum(group["successes"] for group in groups),
        "unavailable": {
            "groups": len(unavailable),
            "mixed_reward_groups": sum(group["mixed_reward"] for group in unavailable),
            "successes": sum(group["successes"] for group in unavailable),
        },
        "group_details": groups,
    }


def infer_workflow_view(row: Mapping[str, Any], task: WorkflowTaskSpec) -> str:
    """Recover a workflow registry view by matching the saved initial environment state."""

    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("episode conversations must be a list")
    initial_state: dict[str, Any] | None = None
    for message in conversations[2:]:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            candidate = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("environment_schema") == (
            "registry-grounded-rl/stateful-workflow-env-v2"
        ):
            initial_state = candidate
            break
    if initial_state is None:
        raise ValueError("workflow initial state is absent from trajectory")
    matches = [
        view.value
        for view in RegistryView
        if RegistryWorkflowEnv(task, view).reset() == initial_state
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one matching registry view, found {matches}")
    return matches[0]


def analyze_episode(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract success, action, and environment-error evidence from one trajectory."""

    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("episode conversations must be a list")
    actions: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    environment_errors: list[str] = []
    for message in conversations[2:]:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str):
            continue
        if role == "assistant":
            parsed = parse_agentgym_action(content)
            if parsed.action is None:
                parse_errors.append(parsed.parse_error or "unknown_parse_error")
            else:
                actions.append(parsed.action)
        elif role == "user":
            try:
                state = json.loads(content)
            except json.JSONDecodeError:
                continue
            if not isinstance(state, Mapping):
                continue
            observation = state.get("observation")
            if isinstance(observation, Mapping) and observation.get("error"):
                environment_errors.append(str(observation["error"]))
            if state.get("parse_error"):
                environment_errors.append(f"parse_error:{state['parse_error']}")
    action_types = [str(action["type"]) for action in actions]
    return {
        "item_id": int(row["item_id"]),
        "success": float(row["reward"]) == 1.0,
        "reward": float(row["reward"]),
        "action_count": len(actions) + len(parse_errors),
        "parsed_action_count": len(actions),
        "parse_error_count": len(parse_errors),
        "tool_call_count": action_types.count("tool_call"),
        "first_action_type": action_types[0] if action_types else "parse_error",
        "terminal_action_type": action_types[-1] if action_types else "parse_error",
        "used_unavailable": "unavailable" in action_types,
        "environment_error_count": len(environment_errors),
        "parse_errors": parse_errors,
        "environment_errors": environment_errors,
    }


def summarize_cell(run_dir: Path, *, expected_episodes: int = 64) -> dict[str, Any]:
    rows = load_trajectory_rows(run_dir)
    episodes = [analyze_episode(row) for row in rows]
    item_ids = [episode["item_id"] for episode in episodes]
    if len(episodes) != expected_episodes:
        raise ValueError(f"{run_dir}: expected {expected_episodes} episodes, found {len(episodes)}")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError(f"{run_dir}: duplicate item IDs")
    successes = sum(episode["success"] for episode in episodes)
    first_actions = Counter(episode["first_action_type"] for episode in episodes)
    terminal_actions = Counter(episode["terminal_action_type"] for episode in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes),
        "wilson_ci95": wilson_interval(successes, len(episodes)),
        "mean_actions": sum(episode["action_count"] for episode in episodes) / len(episodes),
        "mean_tool_calls": sum(episode["tool_call_count"] for episode in episodes) / len(episodes),
        "episodes_with_parse_error": sum(episode["parse_error_count"] > 0 for episode in episodes),
        "episodes_with_environment_error": sum(
            episode["environment_error_count"] > 0 for episode in episodes
        ),
        "episodes_using_unavailable": sum(episode["used_unavailable"] for episode in episodes),
        "first_action_types": dict(sorted(first_actions.items())),
        "terminal_action_types": dict(sorted(terminal_actions.items())),
        "outcomes": {str(episode["item_id"]): episode["success"] for episode in episodes},
    }


def flatten_outcomes(cells: Mapping[str, Mapping[str, Any]], views: Iterable[str]) -> list[bool]:
    outcomes: list[bool] = []
    for view in views:
        outcomes.extend(
            bool(value)
            for _, value in sorted(cells[view]["outcomes"].items(), key=lambda x: int(x[0]))
        )
    return outcomes


def task_outcomes(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[int, bool]]:
    return {
        view: {int(item_id): bool(value) for item_id, value in cell["outcomes"].items()}
        for view, cell in cells.items()
    }


def aggregate_cells(cells: Mapping[str, Mapping[str, Any]], views: Sequence[str]) -> dict[str, Any]:
    episodes = sum(int(cells[view]["episodes"]) for view in views)
    successes = sum(int(cells[view]["successes"]) for view in views)
    outcomes_by_item = task_outcomes(cells)
    item_ids = sorted(set.intersection(*(set(outcomes_by_item[view]) for view in views)))
    all_view_successes = sum(
        all(outcomes_by_item[view][item_id] for view in views) for item_id in item_ids
    )
    return {
        "views": list(views),
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes,
        "wilson_ci95": wilson_interval(successes, episodes),
        "tasks": len(item_ids),
        "all_views_successes": all_view_successes,
        "all_views_success_rate": all_view_successes / len(item_ids),
        "mean_views_succeeded_per_task": successes / len(item_ids),
    }
