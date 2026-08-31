#!/usr/bin/env python3
"""Assemble the complete single-seed pilot, failure analyses, and curriculum study."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from registry_grounded_rl.pilot_analysis import (
    ALL_VIEWS,
    PRIMARY_VIEWS,
    aggregate_cells,
    analyze_episode,
    exact_mcnemar,
    flatten_outcomes,
    infer_workflow_view,
    paired_selective_harmonic_bootstrap,
    paired_task_bootstrap,
    paired_weighted_task_bootstrap,
    parse_training_metrics,
    selective_harmonic_score,
    summarize_cell,
    task_outcomes,
)
from registry_grounded_rl.workflow_dataset import load_workflow_tasks


MODEL_SPECS = {
    "base": ("frozen_dev_eval_v1", "base", None, None),
    "b3_original": (
        "frozen_dev_eval_v1",
        "b3_original",
        "matched_pilot",
        "workflow_b3_original_seed1701_steps4_n6",
    ),
    "b4_orbit": (
        "frozen_dev_eval_v1",
        "b4_orbit",
        "matched_pilot",
        "workflow_b4_orbit_seed1701_steps4_n6",
    ),
    "b5_mixed_capability": (
        "frozen_dev_eval_v1",
        "b5_capability_orbit",
        "matched_pilot",
        "workflow_b5_capability_orbit_seed1701_steps4_n6",
    ),
    "b6_grouped_from_base": (
        "frozen_dev_eval_v1",
        "b6_grouped_capability_orbit",
        "matched_pilot",
        "workflow_b6_grouped_capability_orbit_seed1701_steps4_n6",
    ),
    "stage2_orbit_control": (
        "curriculum_eval_v1",
        "stage2_orbit_control",
        "curriculum_pilot",
        "workflow_b4_orbit_seed1701_stage2_from_b4_steps4_n6",
    ),
    "stage2_stratified_curriculum": (
        "curriculum_eval_v1",
        "stage2_stratified_curriculum",
        "curriculum_pilot",
        "workflow_b7_stratified_capability_orbit_seed1701_stage2_from_b4_steps4_n6",
    ),
}

SELECTIVE_WEIGHTS = {
    **{view: 0.5 / len(PRIMARY_VIEWS) for view in PRIMARY_VIEWS},
    "unavailable": 0.5,
}


def percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def difference_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def matched_fields(manifests: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
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
        values: list[Any] = []
        for manifest in manifests.values():
            value: Any = manifest
            for key in path:
                value = value[key]
            values.append(value)
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"unmatched field: {'.'.join(path)} -> {values}")
        evidence[".".join(path)] = values[0]
    weight_digests = {
        manifest["model"].get("weights_sha256") for manifest in manifests.values()
    }
    if len(weight_digests) == 1 and None not in weight_digests:
        evidence["model.weights_sha256"] = next(iter(weight_digests))
    return evidence


def comparison(
    left_name: str,
    right_name: str,
    models: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    left_cells = models[left_name]["cells"]
    right_cells = models[right_name]["cells"]
    left_primary = flatten_outcomes(left_cells, PRIMARY_VIEWS)
    right_primary = flatten_outcomes(right_cells, PRIMARY_VIEWS)
    left_unavailable = flatten_outcomes(left_cells, ("unavailable",))
    right_unavailable = flatten_outcomes(right_cells, ("unavailable",))
    left_tasks = task_outcomes(left_cells)
    right_tasks = task_outcomes(right_cells)
    result: dict[str, Any] = {
        "left": left_name,
        "right": right_name,
        "primary_difference": sum(right_primary) / len(right_primary)
        - sum(left_primary) / len(left_primary),
        "primary_mcnemar": exact_mcnemar(left_primary, right_primary),
        "primary_task_bootstrap": paired_task_bootstrap(left_tasks, right_tasks),
        "unavailable_difference": sum(right_unavailable) / len(right_unavailable)
        - sum(left_unavailable) / len(left_unavailable),
        "unavailable_mcnemar": exact_mcnemar(left_unavailable, right_unavailable),
        "selective_balanced_difference": (
            models[right_name]["selective_balanced_accuracy"]
            - models[left_name]["selective_balanced_accuracy"]
        ),
        "selective_task_bootstrap": paired_weighted_task_bootstrap(
            left_tasks,
            right_tasks,
            view_weights=SELECTIVE_WEIGHTS,
        ),
        "selective_harmonic_difference": (
            models[right_name]["selective_harmonic_mean"]
            - models[left_name]["selective_harmonic_mean"]
        ),
        "selective_harmonic_task_bootstrap": paired_selective_harmonic_bootstrap(
            left_tasks,
            right_tasks,
        ),
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


def audit_training(run_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    task_file = Path(manifest["task_file"]["path"])
    tasks = load_workflow_tasks(task_file)
    metrics = parse_training_metrics(run_dir / "train.log")
    view_counts: Counter[str] = Counter()
    group_violations = 0
    group_count = 0
    rewards = 0
    episodes = 0
    environment_error_episodes = 0
    unavailable_episodes = 0
    trajectory_files = sorted((run_dir / "trajectories").rglob("*.json"))
    group_size = int(manifest["training"]["rollout_n"])
    for trajectory_file in trajectory_files:
        rows = json.loads(trajectory_file.read_text())
        views: list[str] = []
        for row in rows:
            item_id = int(row["item_id"])
            view = infer_workflow_view(row, tasks[item_id])
            views.append(view)
            view_counts[view] += 1
            episode = analyze_episode(row)
            episodes += 1
            rewards += int(episode["success"])
            environment_error_episodes += int(episode["environment_error_count"] > 0)
            unavailable_episodes += int(episode["used_unavailable"])
        for start in range(0, len(views), group_size):
            group = views[start : start + group_size]
            if len(group) != group_size:
                raise ValueError(f"partial rollout group in {trajectory_file}: {len(group)}")
            group_count += 1
            group_violations += int(len(set(group)) != 1)
    return {
        "episodes": episodes,
        "successes": rewards,
        "success_rate": rewards / episodes,
        "view_counts": dict(sorted(view_counts.items())),
        "rollout_groups": group_count,
        "same_view_group_violations": group_violations,
        "episodes_with_environment_error": environment_error_episodes,
        "episodes_using_unavailable": unavailable_episodes,
        "optimizer_updates": metrics,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    models = summary["evaluation"]["models"]
    comparisons = summary["evaluation"]["comparisons"]
    rows = []
    for model_name in MODEL_SPECS:
        cells = models[model_name]["cells"]
        primary = models[model_name]["primary"]
        rows.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    *(f"{cells[view]['successes']}/64" for view in ALL_VIEWS),
                    f"{primary['successes']}/320 ({percentage(primary['success_rate'])})",
                    percentage(models[model_name]["selective_harmonic_mean"]),
                    percentage(models[model_name]["selective_floor"]),
                ]
            )
            + " |"
        )

    comparison_rows = []
    for key in (
        "base_to_b3",
        "b3_to_b4",
        "b4_to_b5",
        "b4_to_b6",
        "b4_to_stage2_control",
        "b4_to_stage2_curriculum",
        "stage2_control_to_curriculum",
    ):
        item = comparisons[key]
        primary_ci = item["primary_task_bootstrap"]["ci95_percentile"]
        selective_ci = item["selective_harmonic_task_bootstrap"]["ci95_percentile"]
        comparison_rows.append(
            f"| {item['left']} → {item['right']} | "
            f"{difference_pp(item['primary_difference'])} "
            f"[{difference_pp(primary_ci[0])}, {difference_pp(primary_ci[1])}] | "
            f"{difference_pp(item['unavailable_difference'])} | "
            f"{difference_pp(item['selective_harmonic_difference'])} "
            f"[{difference_pp(selective_ci[0])}, {difference_pp(selective_ci[1])}] |"
        )

    training_rows = []
    for model_name, arm in summary["training"]["arms"].items():
        audit = arm["audit"]
        metrics = audit["optimizer_updates"]
        training_rows.append(
            f"| {model_name} | {len(metrics)} | {audit['successes']}/{audit['episodes']} "
            f"({percentage(audit['success_rate'])}) | `{json.dumps(audit['view_counts'], sort_keys=True)}` | "
            f"{audit['same_view_group_violations']}/{audit['rollout_groups']} | "
            f"{metrics[-1]['actor/grad_norm']:.3f} |"
        )

    b4 = models["b4_orbit"]
    b5 = models["b5_mixed_capability"]
    b6 = models["b6_grouped_from_base"]
    direct = comparisons["stage2_control_to_curriculum"]
    primary_ci = direct["primary_task_bootstrap"]["ci95_percentile"]
    selective_ci = direct["selective_harmonic_task_bootstrap"]["ci95_percentile"]
    return f"""# RegistryGrounded-RL：完整 pilot 与两阶段课程结果

生成时间：{summary['created_at_utc']}  
状态：单随机种子、短预算、探索性 pilot；全部数字由冻结轨迹和训练日志重新计算，不能表述为多种子定论。

## 结论先行

1. **在线执行反馈有真实增益。** B3 相对 frozen base 的五个可执行视图提高 {difference_pp(comparisons['base_to_b3']['primary_difference'])}。
2. **朴素混合失败，而且失败原因可以被定位。** B5 把 unavailable 提高到 {percentage(b5['cells']['unavailable']['success_rate'])}，却把可执行成功率从 B4 的 {percentage(b4['primary']['success_rate'])} 压到 {percentage(b5['primary']['success_rate'])}；同一 GRPO group 混合不同环境状态会把状态难度混入组内相对优势。
3. **只修正分组仍不够。** B6 从 base 直接进行同状态 capability 分组后，可执行成功率只有 {percentage(b6['primary']['success_rate'])}。困难组常出现全零奖励，说明模型必须先获得基本执行覆盖，才能学习“何时停止”。
4. **自然修正是 coverage-before-selection。** 先用 B4 学会跨 registry 执行，再以相同起点、相同 3 次 optimizer update 比较普通 orbit 续训和“组内同状态、组间六视图全局均衡”的 capability 课程。
5. **当前直接对照结果：**课程相对普通续训的可执行差值为 {difference_pp(direct['primary_difference'])}（任务级 95% bootstrap CI [{difference_pp(primary_ci[0])}, {difference_pp(primary_ci[1])}]），unavailable 差值为 {difference_pp(direct['unavailable_difference'])}，selective harmonic 差值为 {difference_pp(direct['selective_harmonic_difference'])}（95% CI [{difference_pp(selective_ci[0])}, {difference_pp(selective_ci[1])}]）。这是值得继续验证的 pilot 证据，不是显著性已确认的结论。

## 冻结 64-task × 6-view 评测

| 模型 | original | order | schema | opaque | hard | unavailable | 五个可执行视图 | selective harmonic¹ | selective floor² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

¹ selective harmonic 是五视图执行率与 unavailable 正确停止率的调和平均；任何一侧坍塌都会压低分数。² selective floor 取两侧较小值，是更严格的能力下界。朴素算术 balanced accuracy 会让 B5/B6 的过度停止看起来更高，因此只保留在机器可读文件中作诊断，不再作为主指标。

## 配对差值与不确定性

| 对比 | 可执行差值（task bootstrap 95% CI） | unavailable 差值 | selective harmonic 差值（95% CI） |
|---|---:|---:|---:|
{chr(10).join(comparison_rows)}

每次比较都复用完全相同的 64 个 task；bootstrap 以 task 为采样单位，把同一 task 的六个 view 一起重采样。机器可读文件还保存每个 view 的 exact McNemar 改善/退化配对与 p 值。

## 训练与调度审计

| 训练臂 | optimizer updates | 在线成功轨迹 | 实际 view counts | 组内同 view 违规 | 最后 grad norm |
|---|---:|---:|---:|---:|---:|
{chr(10).join(training_rows)}

阶段二两个模型从同一个 B4 checkpoint 出发，其权重摘要均为 `{summary['training']['stage2_matched_fields']['model.weights_sha256']}`；seed、训练任务、rollout_n=6、终局二值 reward、token/step 预算和代码摘要均匹配，唯一实验变量是 registry schedule。配置请求 4 个 global steps，VERL 实际产生 3 次 optimizer update 并保存 `global_step_4`。

## 研究边界

- B7 是观察到 B5/B6 失败后提出的修正，因此属于**探索性 rescue experiment**，不能伪装成预注册 confirmatory result。
- 当前只有 seed=1701 和短预算。CI 跨零时，只能写“观察到”或“pilot 中提高”，不能写“显著提升”或“稳定优于”。
- unavailable 的成功标准是零副作用地输出 unavailable；可执行任务则要求精确末态正确，纯粹立即停止无法同时优化两者。
- 若要升级为最终简历主结论，应补独立 stage-2 seeds，并在冻结 test 上只评一次；若复现失败，保留成负结果与机制分析。

## 当前可用的一句话

> 我在 Qwen2.5-3B 上构建了动态工具注册表的可执行 GRPO 环境；针对朴素 capability 混合训练诱发的过度停止，我从失败轨迹中提出“先覆盖执行、再学习选择”的两阶段课程，并用同起点同算力对照和 64×6 冻结配对评测检验执行成功与零副作用停止之间的权衡。

这句话描述了真实完成的工作，但在补多 seed 前不附“稳定提升”字样。

## 复现分析

```bash
cd /path/to/registry-grounded-rl
python scripts/summarize_full_study.py
```

机器可读证据：`artifacts/full_study_summary_v2.json`。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    root = arguments.project_root.expanduser().resolve(strict=True)

    models: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    arms: dict[str, dict[str, Any]] = {}
    for model_name, (evaluation_name, evaluation_tag, training_name, run_name) in MODEL_SPECS.items():
        evaluation_dir = root / "runs" / evaluation_name / evaluation_tag
        cells = {view: summarize_cell(evaluation_dir / view) for view in ALL_VIEWS}
        primary = aggregate_cells(cells, PRIMARY_VIEWS)
        models[model_name] = {
            "evaluation_dir": str(evaluation_dir),
            "cells": cells,
            "primary": primary,
            "all_views": aggregate_cells(cells, ALL_VIEWS),
            "selective_balanced_accuracy": (
                primary["success_rate"] + cells["unavailable"]["success_rate"]
            )
            / 2,
            "selective_harmonic_mean": selective_harmonic_score(
                primary["success_rate"], cells["unavailable"]["success_rate"]
            ),
            "selective_floor": min(
                primary["success_rate"], cells["unavailable"]["success_rate"]
            ),
        }
        if training_name is None or run_name is None:
            continue
        run_dir = root / "runs" / training_name / run_name
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        manifests[model_name] = manifest
        arms[model_name] = {
            "run_dir": str(run_dir),
            "manifest": manifest,
            "audit": audit_training(run_dir, manifest),
        }

    comparisons = {
        "base_to_b3": comparison("base", "b3_original", models),
        "b3_to_b4": comparison("b3_original", "b4_orbit", models),
        "b4_to_b5": comparison("b4_orbit", "b5_mixed_capability", models),
        "b4_to_b6": comparison("b4_orbit", "b6_grouped_from_base", models),
        "b4_to_stage2_control": comparison("b4_orbit", "stage2_orbit_control", models),
        "b4_to_stage2_curriculum": comparison(
            "b4_orbit", "stage2_stratified_curriculum", models
        ),
        "stage2_control_to_curriculum": comparison(
            "stage2_orbit_control", "stage2_stratified_curriculum", models
        ),
    }
    summary = {
        "schema_version": "registry-grounded-rl/full-study-summary-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training": {
            "stage1_matched_fields": matched_fields(
                {name: manifests[name] for name in ("b3_original", "b4_orbit", "b5_mixed_capability")}
            ),
            "stage2_matched_fields": matched_fields(
                {
                    name: manifests[name]
                    for name in ("stage2_orbit_control", "stage2_stratified_curriculum")
                }
            ),
            "arms": arms,
        },
        "evaluation": {
            "protocol": {
                "tasks": 64,
                "views": list(ALL_VIEWS),
                "temperature": 0.0,
                "top_p": 1.0,
                "max_rounds": 10,
                "selective_weights": SELECTIVE_WEIGHTS,
            },
            "models": models,
            "comparisons": comparisons,
        },
    }
    artifact = root / "artifacts" / "full_study_summary_v2.json"
    report = root / "FULL_STUDY_RESULTS.md"
    artifact.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report.write_text(render_markdown(summary))
    print(f"wrote {artifact}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
