#!/usr/bin/env python3
"""Build machine-readable and human-readable evidence for the matched pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from registry_grounded_rl.pilot_analysis import (
    ALL_VIEWS,
    PRIMARY_VIEWS,
    aggregate_cells,
    exact_mcnemar,
    flatten_outcomes,
    paired_task_bootstrap,
    parse_training_metrics,
    summarize_cell,
    task_outcomes,
)


MODEL_RUNS = {
    "base": None,
    "b3_original": "workflow_b3_original_seed1701_steps4_n6",
    "b4_orbit": "workflow_b4_orbit_seed1701_steps4_n6",
    "b5_capability_orbit": "workflow_b5_capability_orbit_seed1701_steps4_n6",
}


def percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def difference_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def comparison(
    left_name: str,
    right_name: str,
    models: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    left_cells = models[left_name]["cells"]
    right_cells = models[right_name]["cells"]
    left_primary = flatten_outcomes(left_cells, PRIMARY_VIEWS)
    right_primary = flatten_outcomes(right_cells, PRIMARY_VIEWS)
    left_tasks = task_outcomes(left_cells)
    right_tasks = task_outcomes(right_cells)
    result: dict[str, Any] = {
        "left": left_name,
        "right": right_name,
        "primary_difference": sum(right_primary) / len(right_primary)
        - sum(left_primary) / len(left_primary),
        "primary_mcnemar": exact_mcnemar(left_primary, right_primary),
        "primary_task_bootstrap": paired_task_bootstrap(left_tasks, right_tasks),
        "by_view": {},
    }
    for view in ALL_VIEWS:
        before = flatten_outcomes(left_cells, (view,))
        after = flatten_outcomes(right_cells, (view,))
        result["by_view"][view] = {
            "difference": sum(after) / len(after) - sum(before) / len(before),
            "mcnemar": exact_mcnemar(before, after),
        }
    return result


def matched_fields(manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    paths = (
        ("benchmark",),
        ("run_seed",),
        ("model", "config_sha256"),
        ("model", "path"),
        ("index_file", "sha256"),
        ("task_file", "sha256"),
        ("source", "code_sha256"),
        ("training", "algorithm"),
        ("training", "max_rounds"),
        ("training", "n_gpus"),
        ("training", "rollout_n"),
        ("training", "terminal_reward"),
        ("training", "total_epochs"),
        ("training", "total_training_steps"),
        ("training", "train_batch_size"),
    )
    evidence: dict[str, Any] = {}
    for path in paths:
        values = []
        for manifest in manifests.values():
            value: Any = manifest
            for key in path:
                value = value[key]
            values.append(value)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"unmatched pilot field: {'.'.join(path)} -> {values}")
        evidence[".".join(path)] = values[0]
    return evidence


def render_markdown(summary: dict[str, Any]) -> str:
    models = summary["evaluation"]["models"]
    comparisons = summary["evaluation"]["comparisons"]
    rows = []
    for model_name in MODEL_RUNS:
        cells = models[model_name]["cells"]
        primary = models[model_name]["primary"]
        selective = models[model_name]["selective_balanced_accuracy"]
        rows.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    *(
                        f"{cells[view]['successes']}/64 ({percentage(cells[view]['success_rate'])})"
                        for view in ALL_VIEWS
                    ),
                    f"{primary['successes']}/320 ({percentage(primary['success_rate'])})",
                    percentage(selective),
                ]
            )
            + " |"
        )

    comparison_rows = []
    for key in ("base_to_b3", "base_to_b4", "b3_to_b4", "b4_to_b5"):
        item = comparisons[key]
        mcnemar = item["primary_mcnemar"]
        bootstrap = item["primary_task_bootstrap"]
        low, high = bootstrap["ci95_percentile"]
        comparison_rows.append(
            f"| {item['left']} → {item['right']} | {difference_pp(item['primary_difference'])} | "
            f"{mcnemar['improved']} / {mcnemar['regressed']} | {mcnemar['p_value_two_sided']:.4f} | "
            f"[{difference_pp(low)}, {difference_pp(high)}] |"
        )

    behavior_rows = []
    for model_name in MODEL_RUNS:
        primary_cells = [models[model_name]["cells"][view] for view in PRIMARY_VIEWS]
        unavailable = models[model_name]["cells"]["unavailable"]
        solvable_unavailable = sum(cell["episodes_using_unavailable"] for cell in primary_cells)
        solvable_errors = sum(cell["episodes_with_environment_error"] for cell in primary_cells)
        behavior_rows.append(
            f"| {model_name} | {solvable_unavailable}/320 | {solvable_errors}/320 | "
            f"{unavailable['episodes_using_unavailable']}/64 | {unavailable['successes']}/64 |"
        )

    training_rows = []
    for arm, training in summary["training"]["arms"].items():
        metrics = training["optimizer_updates"]
        successes = sum(round(item["critic/score/mean"] * 96) for item in metrics)
        total = 96 * len(metrics)
        training_rows.append(
            f"| {arm} | {len(metrics)} | {successes}/{total} ({percentage(successes / total)}) | "
            f"{metrics[-1]['actor/grad_norm']:.3f} | {metrics[-1]['critic/task_round/mean']:.3f} |"
        )

    b3 = models["b3_original"]["primary"]["success_rate"]
    b4 = models["b4_orbit"]["primary"]["success_rate"]
    b5 = models["b5_capability_orbit"]
    base = models["base"]["primary"]["success_rate"]
    return f"""# RegistryGrounded-RL：同预算 GRPO 匹配试验结果

生成时间：{summary["created_at_utc"]}  
状态：单随机种子短预算 pilot；所有数字来自保存的训练日志与冻结轨迹，由脚本重新计算。

## 结论先行

1. **执行反馈本身有效。** 原始视图 GRPO（B3）在五个可执行冻结视图上由 base 的 {percentage(base)} 提升到 {percentage(b3)}，即 {difference_pp(b3 - base)}。
2. **registry orbit 有额外但尚不确定的增益。** B4 达到 {percentage(b4)}，比 B3 高 {difference_pp(b4 - b3)}；任务级配对 bootstrap 95% 区间跨过 0，因此当前不能把它写成稳定显著提升。
3. **直接混入 unavailable 会产生失败模式。** B5 的 unavailable 正确率达到 {percentage(b5["cells"]["unavailable"]["success_rate"])}，但可执行视图降到 {percentage(b5["primary"]["success_rate"])}。这不是理想的 selective execution，而是明显的停止捷径。
4. **最有价值的研究问题被实验逼出来了：** GRPO 的同组候选若来自不同 registry 状态，容易把“状态难度”误当成“动作质量”。下一步应保持每个 GRPO group 内环境状态一致，只在 group 之间轮换 registry view，再检验能否同时保住执行能力与不可用判断。

## 冻结评测主表

| 模型 | original | order | schema surface | opaque alias | hard distractor | unavailable | 五个可执行视图 | selective balanced¹ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

¹ selective balanced =（五个可执行视图成功率 + unavailable 正确率）/ 2；它避免 unavailable 只占六分之一而被平均数掩盖。

## 配对统计

| 对比 | 五视图差值 | 改善 / 退化配对 | exact McNemar p | 任务级 paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
{chr(10).join(comparison_rows)}

McNemar 把每个 task-view 当作配对二值结果；bootstrap 则以 task 为采样单位，同时保留同一任务五个视图的相关性。单 seed pilot 的区间用于约束表述，不等于最终显著性结论。

## 行为审计

| 模型 | 可执行视图中使用 unavailable | 可执行视图中出现环境/解析错误 | unavailable 视图中使用 unavailable | unavailable 成功 |
|---|---:|---:|---:|---:|
{chr(10).join(behavior_rows)}

B5 的高 unavailable 分数和可执行能力坍塌同时出现，说明二值奖励下的短路径“立即停止”压过了多步工具执行。后续修正应改训练分组，而不是事后挑指标。

## 训练确实发生的证据

| 训练臂 | optimizer updates | 在线成功轨迹 | 最后 grad norm | 最后平均轮数 |
|---|---:|---:|---:|---:|
{chr(10).join(training_rows)}

三组均从同一个 Qwen2.5-3B-Instruct 起点训练，seed=1701、16 tasks/step、rollout_n=6、单张 NVIDIA A800 80GB、二值终局奖励。配置请求 4 steps；AgentGym-RL/VERL 的 global-step 约定实际执行 3 次优化并保存 `global_step_4`。

## 当前可以与不可以写进简历的内容

可以写：实现动态 registry 的可执行多轮环境、严格状态校验奖励、GRPO 在线训练、同预算消融与冻结轨迹配对评测；发现普通执行 RL 带来跨 schema 的零样本增益，并定位 capability 混合训练导致的过度停止。

暂时不要写：registry orbit 已“显著优于”普通 GRPO；或 selective execution 已经解决。前者单 seed 区间跨 0，后者被 B5 的可执行性能反证。

## 复现

```bash
cd /path/to/registry-grounded-rl
python scripts/summarize_matched_pilot.py
```

机器可读证据见 `artifacts/matched_pilot_summary_v1.json`。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    root = arguments.project_root.expanduser().resolve(strict=True)
    evaluation_root = root / "runs" / "frozen_dev_eval_v1"
    pilot_root = root / "runs" / "matched_pilot"

    models: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_RUNS:
        cells = {view: summarize_cell(evaluation_root / model_name / view) for view in ALL_VIEWS}
        primary = aggregate_cells(cells, PRIMARY_VIEWS)
        all_views = aggregate_cells(cells, ALL_VIEWS)
        models[model_name] = {
            "cells": cells,
            "primary": primary,
            "all_views": all_views,
            "selective_balanced_accuracy": (
                primary["success_rate"] + cells["unavailable"]["success_rate"]
            )
            / 2,
        }

    manifests: dict[str, dict[str, Any]] = {}
    arms: dict[str, dict[str, Any]] = {}
    for model_name, run_name in MODEL_RUNS.items():
        if run_name is None:
            continue
        run_dir = pilot_root / run_name
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        manifests[model_name] = manifest
        arms[model_name] = {
            "run_dir": str(run_dir),
            "manifest": manifest,
            "optimizer_updates": parse_training_metrics(run_dir / "train.log"),
        }

    summary = {
        "schema_version": "registry-grounded-rl/matched-pilot-summary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": {"matched_fields": matched_fields(manifests), "arms": arms},
        "evaluation": {
            "protocol": {
                "tasks": 64,
                "views": list(ALL_VIEWS),
                "temperature": 0.0,
                "top_p": 1.0,
                "max_rounds": 10,
            },
            "models": models,
            "comparisons": {
                "base_to_b3": comparison("base", "b3_original", models),
                "base_to_b4": comparison("base", "b4_orbit", models),
                "b3_to_b4": comparison("b3_original", "b4_orbit", models),
                "b4_to_b5": comparison("b4_orbit", "b5_capability_orbit", models),
            },
        },
    }
    artifact = root / "artifacts" / "matched_pilot_summary_v1.json"
    report = root / "MATCHED_PILOT_RESULTS.md"
    artifact.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report.write_text(render_markdown(summary))
    print(f"wrote {artifact}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
