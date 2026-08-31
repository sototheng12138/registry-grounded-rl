#!/usr/bin/env python3
"""Summarize the independent-sampling selective-execution study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from registry_grounded_rl.pilot_analysis import (
    ALL_VIEWS,
    PRIMARY_VIEWS,
    aggregate_cells,
    exact_mcnemar,
    flatten_outcomes,
    hierarchical_paired_bootstrap,
    paired_selective_harmonic_bootstrap,
    paired_task_bootstrap,
    parse_training_metrics,
    selective_harmonic_score,
    summarize_cell,
    task_outcomes,
)


MODEL_SPECS = {
    "b4_start": ("frozen_dev_eval_v1", "b4_orbit"),
    "b8_seed2901": ("selective_screen_eval_v1", "b8_seed2901_lr5e7_gs2"),
    "b8_seed2911": ("selective_confirmatory_eval_v1", "b8_seed2911_lr5e7_gs2"),
    "b8_seed2921": ("selective_confirmatory_eval_v1", "b8_seed2921_lr5e7_gs2"),
    "b9_seed2911": ("selective_confirmatory_eval_v1", "b9_seed2911_lr5e7_gs2"),
}

B8_NAMES = ("b8_seed2901", "b8_seed2911", "b8_seed2921")
FINAL_CODE_B8_NAMES = ("b8_seed2911", "b8_seed2921")

TRAINING_SPECS = {
    "b8_seed2901": (
        "selective_matched_v1",
        "workflow_b8_selective_capability_orbit_seed2901_stage2_from_b4_indseed_lr5e7_kl005_ent0_steps3_n6",
        "selective_matched_b8_seed2901_rollout_audit.json",
    ),
    "b8_seed2911": (
        "selective_confirmatory_v1",
        "workflow_b8_selective_capability_orbit_seed2911_stage2_from_b4_indseed_lr5e7_kl005_ent0_steps3_n6",
        "selective_confirmatory_b8_seed2911_rollout_audit.json",
    ),
    "b8_seed2921": (
        "selective_confirmatory_v1",
        "workflow_b8_selective_capability_orbit_seed2921_stage2_from_b4_indseed_lr5e7_kl005_ent0_steps3_n6",
        "selective_confirmatory_b8_seed2921_rollout_audit.json",
    ),
    "b9_seed2911": (
        "selective_confirmatory_v1",
        "workflow_b9_stratified_solvable_orbit_seed2911_stage2_from_b4_indseed_lr5e7_kl005_ent0_steps3_n6",
        "selective_confirmatory_b9_seed2911_rollout_audit.json",
    ),
}


def percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def difference_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def assert_equal_fields(
    manifests: Mapping[str, Mapping[str, Any]],
    paths: Sequence[Sequence[str]],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for path in paths:
        values = [nested_value(manifest, path) for manifest in manifests.values()]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"unmatched field {'.'.join(path)}: {values}")
        evidence[".".join(path)] = values[0]
    return evidence


def load_model(root: Path, evaluation_name: str, tag: str) -> dict[str, Any]:
    evaluation_dir = root / "runs" / evaluation_name / tag
    cells = {view: summarize_cell(evaluation_dir / view) for view in ALL_VIEWS}
    primary = aggregate_cells(cells, PRIMARY_VIEWS)
    unavailable = cells["unavailable"]
    eval_manifests = {
        view: json.loads((evaluation_dir / view / "eval_manifest.json").read_text())
        for view in ALL_VIEWS
    }
    eval_protocol = assert_equal_fields(
        eval_manifests,
        (
            ("model", "sha256"),
            ("task_file", "sha256"),
            ("eval_data", "manifest", "source_sha256"),
            ("seed",),
            ("decoding", "temperature"),
            ("decoding", "top_p"),
            ("decoding", "max_tokens_per_turn"),
            ("max_rounds",),
        ),
    )
    return {
        "evaluation_dir": str(evaluation_dir),
        "cells": cells,
        "primary": primary,
        "unavailable": unavailable,
        "selective_balanced_accuracy": (
            primary["success_rate"] + unavailable["success_rate"]
        )
        / 2,
        "selective_harmonic_mean": selective_harmonic_score(
            primary["success_rate"], unavailable["success_rate"]
        ),
        "eval_protocol": eval_protocol,
    }


def compare_models(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    left_tasks = task_outcomes(left["cells"])
    right_tasks = task_outcomes(right["cells"])
    primary_left = flatten_outcomes(left["cells"], PRIMARY_VIEWS)
    primary_right = flatten_outcomes(right["cells"], PRIMARY_VIEWS)
    unavailable_left = flatten_outcomes(left["cells"], ("unavailable",))
    unavailable_right = flatten_outcomes(right["cells"], ("unavailable",))
    return {
        "primary_difference": right["primary"]["success_rate"]
        - left["primary"]["success_rate"],
        "primary_mcnemar": exact_mcnemar(primary_left, primary_right),
        "primary_task_bootstrap": paired_task_bootstrap(
            left_tasks,
            right_tasks,
            samples=20_000,
            seed=bootstrap_seed,
        ),
        "unavailable_difference": right["unavailable"]["success_rate"]
        - left["unavailable"]["success_rate"],
        "unavailable_mcnemar": exact_mcnemar(unavailable_left, unavailable_right),
        "selective_harmonic_difference": right["selective_harmonic_mean"]
        - left["selective_harmonic_mean"],
        "selective_harmonic_task_bootstrap": paired_selective_harmonic_bootstrap(
            left_tasks,
            right_tasks,
            samples=20_000,
            seed=bootstrap_seed,
        ),
        "by_view": {
            view: {
                "difference": right["cells"][view]["success_rate"]
                - left["cells"][view]["success_rate"],
                "mcnemar": exact_mcnemar(
                    flatten_outcomes(left["cells"], (view,)),
                    flatten_outcomes(right["cells"], (view,)),
                ),
            }
            for view in ALL_VIEWS
        },
    }


def mean_metric(models: Mapping[str, Mapping[str, Any]], names: Sequence[str], key: str) -> float:
    return sum(float(models[name][key]) for name in names) / len(names)


def aggregate_b8(models: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> dict[str, Any]:
    return {
        "training_seeds": [int(name.removeprefix("b8_seed")) for name in names],
        "mean_primary_success_rate": sum(
            models[name]["primary"]["success_rate"] for name in names
        )
        / len(names),
        "mean_unavailable_success_rate": sum(
            models[name]["unavailable"]["success_rate"] for name in names
        )
        / len(names),
        "mean_selective_balanced_accuracy": mean_metric(
            models, names, "selective_balanced_accuracy"
        ),
        "mean_selective_harmonic_mean": mean_metric(
            models, names, "selective_harmonic_mean"
        ),
        "mean_view_success_rate": {
            view: sum(models[name]["cells"][view]["success_rate"] for name in names)
            / len(names)
            for view in ALL_VIEWS
        },
    }


def training_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for name, (run_root, run_name, audit_name) in TRAINING_SPECS.items():
        run_dir = root / "runs" / run_root / run_name
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        audit = json.loads((root / "artifacts" / audit_name).read_text())
        manifests[name] = manifest
        evidence[name] = {
            "run_dir": str(run_dir),
            "manifest": manifest,
            "optimizer_updates": parse_training_metrics(run_dir / "train.log"),
            "rollout_group_audit": audit,
        }

    common_paths = (
        ("benchmark",),
        ("model", "config_sha256"),
        ("model", "weights_sha256"),
        ("index_file", "sha256"),
        ("task_file", "sha256"),
        ("training", "algorithm"),
        ("training", "actor_lr"),
        ("training", "entropy_coeff"),
        ("training", "kl_loss_coef"),
        ("training", "max_rounds"),
        ("training", "n_gpus"),
        ("training", "rollout_n"),
        ("training", "sampling_seed_strategy"),
        ("training", "terminal_reward"),
        ("training", "total_epochs"),
        ("training", "total_training_steps"),
        ("training", "train_batch_size"),
    )
    common = assert_equal_fields(manifests, common_paths)
    final_code = assert_equal_fields(
        {name: manifests[name] for name in ("b8_seed2911", "b8_seed2921", "b9_seed2911")},
        (*common_paths, ("source", "code_sha256"), ("source", "agentgym_rollout_sha256")),
    )
    seed2911 = assert_equal_fields(
        {name: manifests[name] for name in ("b8_seed2911", "b9_seed2911")},
        (*common_paths, ("source", "code_sha256"), ("run_seed",), ("training", "data_seed"),
         ("training", "rollout_seed")),
    )
    checks = {
        "all_stage2_common_fields": common,
        "final_code_common_fields": final_code,
        "seed2911_b8_b9_matched_fields": seed2911,
        "exploratory_seed2901_code_sha256": manifests["b8_seed2901"]["source"]["code_sha256"],
        "final_code_sha256": manifests["b8_seed2911"]["source"]["code_sha256"],
        "seed2901_is_exploratory_due_to_additive_source_change": (
            manifests["b8_seed2901"]["source"]["code_sha256"]
            != manifests["b8_seed2911"]["source"]["code_sha256"]
        ),
    }
    return evidence, checks


def render_markdown(summary: Mapping[str, Any]) -> str:
    models = summary["evaluation"]["models"]
    all_seeds = summary["evaluation"]["b8_all_three_seeds"]
    hierarchical_all = summary["evaluation"]["b8_vs_b4_hierarchical_all_three"]
    hierarchical_final = summary["evaluation"]["b8_vs_b4_hierarchical_final_code_two"]
    b8_b9 = summary["evaluation"]["b8_vs_b9_seed2911"]
    b4 = models["b4_start"]

    table_rows = []
    for name in MODEL_SPECS:
        model = models[name]
        cells = model["cells"]
        table_rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    *(f"{cells[view]['successes']}/64" for view in ALL_VIEWS),
                    f"{model['primary']['successes']}/320 ({percentage(model['primary']['success_rate'])})",
                    percentage(model["selective_harmonic_mean"]),
                ]
            )
            + " |"
        )

    per_seed_rows = []
    for name in B8_NAMES:
        comparison = summary["evaluation"]["b8_vs_b4_per_seed"][name]
        primary_ci = comparison["primary_task_bootstrap"]["ci95_percentile"]
        harmonic_ci = comparison["selective_harmonic_task_bootstrap"]["ci95_percentile"]
        per_seed_rows.append(
            f"| {name.removeprefix('b8_seed')} | "
            f"{difference_pp(comparison['primary_difference'])} "
            f"[{difference_pp(primary_ci[0])}, {difference_pp(primary_ci[1])}] | "
            f"{difference_pp(comparison['unavailable_difference'])} | "
            f"{difference_pp(comparison['selective_harmonic_difference'])} "
            f"[{difference_pp(harmonic_ci[0])}, {difference_pp(harmonic_ci[1])}] |"
        )

    def hierarchical_row(label: str, result: Mapping[str, Any]) -> str:
        cells = []
        for key in (
            "primary_difference",
            "unavailable_difference",
            "selective_harmonic_difference",
        ):
            metric = result["metrics"][key]
            low, high = metric["ci95_percentile"]
            cells.append(
                f"{difference_pp(metric['difference'])} "
                f"[{difference_pp(low)}, {difference_pp(high)}], "
                f"P(>0)={metric['bootstrap_probability_positive']:.3f}"
            )
        return f"| {label} | " + " | ".join(cells) + " |"

    training_rows = []
    for name, evidence in summary["training"]["runs"].items():
        audit = evidence["rollout_group_audit"]["steps"]
        step_text = "; ".join(
            f"{step}: unique={values['mean_unique_trajectories_per_group']:.2f}, "
            f"mixed={values['mixed_reward_groups']}/16"
            for step, values in audit.items()
        )
        training_rows.append(
            f"| {name} | {len(evidence['optimizer_updates'])} | {step_text} |"
        )

    strict = b8_b9
    strict_primary_ci = strict["primary_task_bootstrap"]["ci95_percentile"]
    strict_harmonic_ci = strict["selective_harmonic_task_bootstrap"]["ci95_percentile"]

    return f"""# RegistryGrounded-RL：独立采样与选择性执行研究结果

生成时间：{summary['created_at_utc']}  
状态：stage-2 冻结 dev 研究已完成；test split 未做模型 rollout，仍保持未触碰。

## 结论先行

1. **先修复训练信号，再讨论课程。** 原 AgentGym-RL 路径给同一 GRPO group 的重复 prompt 复用了同一个采样 seed，旧审计每组平均只有 1.56 条独立轨迹、15/16 组零 reward 方差。项目改成按 `(base seed, global step, round, rank, rollout index)` 派生采样 seed；独立采样审计每组平均提高到 4.12 条，恢复了可学习 advantage。
2. **简单有效的干预是 group-aligned selective curriculum（B8）。** 每个 GRPO group 内保持同一 task 和同一 registry 状态，group 间固定 50:50 轮换可执行与不可执行状态。这样 advantage 比较的是同一环境中的动作质量，不把不同 registry 难度混进组内归一化。
3. **三训练种子相对固定 B4 起点：**五个可执行视图均值由 {percentage(b4['primary']['success_rate'])} 到 {percentage(all_seeds['mean_primary_success_rate'])}（{difference_pp(all_seeds['mean_primary_success_rate'] - b4['primary']['success_rate'])}）；unavailable 由 {percentage(b4['unavailable']['success_rate'])} 到 {percentage(all_seeds['mean_unavailable_success_rate'])}（{difference_pp(all_seeds['mean_unavailable_success_rate'] - b4['unavailable']['success_rate'])}）；执行—停止 harmonic mean 由 {percentage(b4['selective_harmonic_mean'])} 到 {percentage(all_seeds['mean_selective_harmonic_mean'])}（{difference_pp(all_seeds['mean_selective_harmonic_mean'] - b4['selective_harmonic_mean'])}）。
4. **结论有方向性，但不能包装成“稳定显著”。** 第三个 seed 的 unavailable 为 11/64，略低于 B4 的 12/64；三种子层级 bootstrap 的区间见下表。当前可信表述是“平均改善并定位出种子方差”，不是“所有种子都提升”。
5. **同 seed 机制对照支持 capability-boundary exposure 的作用。** seed 2911 下，B8 相比只继续训练 solvable views 的 B9，可执行成功率高 {difference_pp(strict['primary_difference'])}（95% task bootstrap [{difference_pp(strict_primary_ci[0])}, {difference_pp(strict_primary_ci[1])}]），harmonic mean 高 {difference_pp(strict['selective_harmonic_difference'])}（[{difference_pp(strict_harmonic_ci[0])}, {difference_pp(strict_harmonic_ci[1])}]）。B9 的下降伴随大量解析错误，因此 unavailable group 在这里同时是行为边界和短预算稳定器，不能简化成“多加拒绝样本”。

## 冻结评测主表

每个 cell 是同一组 64 个 dev tasks、temperature=0、最多 10 轮。Primary 是前五个可执行视图；harmonic mean 同时惩罚“只会执行”和“只会停止”。

| checkpoint | original | order | schema | opaque | hard | unavailable | primary | harmonic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## B8 相对 B4：逐训练种子

| seed | primary 差值及 task-bootstrap 95% CI | unavailable 差值 | harmonic 差值及 95% CI |
|---:|---:|---:|---:|
{chr(10).join(per_seed_rows)}

seed 2901 是探索性复现：B8 实现与超参数相同，但随后为加入 B9 调度产生了加法式 source hash 变化。seed 2911/2921 使用完全相同的最终代码。checkpoint 选择规则在 seed 2921 运行前已固定为第二次 optimizer update，不按 dev 挑选。

## 跨训练种子的层级 bootstrap

先重采样训练 seed，再在所选 seed 内按 task 成对重采样；这避免把 64 个 task 当作 64 次独立训练。左侧 B4 是固定 stage-1 checkpoint，因此区间衡量的是 stage-2 B8 续训的随机性，不是假装 B4 也有三个训练复现。

| 范围 | primary | unavailable | harmonic |
|---|---:|---:|---:|
{hierarchical_row('全部三 seed（含探索性 2901）', hierarchical_all)}
{hierarchical_row('最终代码两 seed（2911/2921）', hierarchical_final)}

## 训练确实发生且 GRPO 信号非退化

| run | optimizer updates | rollout group 审计 |
|---|---:|---|
{chr(10).join(training_rows)}

四个 stage-2 run 均从同一 B4 权重 `{summary['training']['checks']['all_stage2_common_fields']['model.weights_sha256']}` 开始，使用 Qwen2.5-3B、单张 A800 80GB、batch 16、每 task 6 条 rollout、二值精确末态 reward、学习率 5e-7、KL 0.05、entropy 0，共 2 次实际 optimizer update。seed 2911 的 B8/B9 除 registry schedule 外，代码、起点、数据、算法、预算与随机种子均相同。

## 研究边界

- 这是冻结 **dev** 上的短预算 stage-2 研究，不是外部 benchmark SOTA，也不是 held-out test 结论。
- B8 三 seed 的 primary 总体稳定，但 opaque/schema 的能力分配存在波动；unavailable 也有一个负向 seed。
- B9 是一个 seed 的机制性对照，足以暴露失败模式，但不足以估计 B9 的跨 seed 均值。
- 不声称首创 selective execution、dynamic tools 或 GRPO。项目贡献是把 registry intervention、可执行 verifier、GRPO 分组语义和 rollout RNG 审计连成一条可证伪实验链。

## 当前可用于简历的一句话

> 针对工具 Agent 在 registry 变化或关键能力缺失时盲目执行的问题，我在 Qwen2.5-3B/AgentGym-RL 上构建可验证多轮环境，修复同组 rollout 复用随机种子导致的 GRPO 信号退化，并提出组内同状态、组间 50:50 能力边界课程；三训练种子冻结 dev 上在保持可执行成功率的同时，将正确停止率平均提高 {difference_pp(all_seeds['mean_unavailable_success_rate'] - b4['unavailable']['success_rate'])}，但保留一个负向种子并用层级 bootstrap 报告不确定性。

复现汇总：

```bash
cd /path/to/registry-grounded-rl
python scripts/summarize_selective_study.py
```

机器可读证据：`artifacts/selective_study_summary_v1.json`。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    root = arguments.project_root.expanduser().resolve(strict=True)

    models = {
        name: load_model(root, evaluation_name, tag)
        for name, (evaluation_name, tag) in MODEL_SPECS.items()
    }
    reference_protocol = models["b4_start"]["eval_protocol"]
    for name, model in models.items():
        protocol = model["eval_protocol"]
        for key in (
            "task_file.sha256",
            "eval_data.manifest.source_sha256",
            "seed",
            "decoding.temperature",
            "decoding.top_p",
            "decoding.max_tokens_per_turn",
            "max_rounds",
        ):
            if protocol[key] != reference_protocol[key]:
                raise ValueError(f"evaluation protocol mismatch for {name}: {key}")

    training_runs, training_checks = training_evidence(root)
    b8_vs_b4 = {
        name: compare_models(models["b4_start"], models[name], bootstrap_seed=int(name[-4:]))
        for name in B8_NAMES
    }
    b4_tasks = task_outcomes(models["b4_start"]["cells"])
    all_b8_tasks = [task_outcomes(models[name]["cells"]) for name in B8_NAMES]
    final_b8_tasks = [task_outcomes(models[name]["cells"]) for name in FINAL_CODE_B8_NAMES]
    hierarchical_all = hierarchical_paired_bootstrap(
        [b4_tasks] * len(all_b8_tasks),
        all_b8_tasks,
        samples=50_000,
        seed=3101,
    )
    hierarchical_final = hierarchical_paired_bootstrap(
        [b4_tasks] * len(final_b8_tasks),
        final_b8_tasks,
        samples=50_000,
        seed=3102,
    )

    summary = {
        "schema_version": "registry-grounded-rl/selective-study-summary-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_status": {
            "reported_split": "dev",
            "test_model_rollouts_performed": False,
            "checkpoint_selection": "global_step_2 fixed before seed2921",
        },
        "training": {"runs": training_runs, "checks": training_checks},
        "evaluation": {
            "protocol": {
                "tasks_per_view": 64,
                "views": list(ALL_VIEWS),
                "temperature": 0.0,
                "top_p": 1.0,
                "max_rounds": 10,
            },
            "models": models,
            "b8_all_three_seeds": aggregate_b8(models, B8_NAMES),
            "b8_final_code_two_seeds": aggregate_b8(models, FINAL_CODE_B8_NAMES),
            "b8_vs_b4_per_seed": b8_vs_b4,
            "b8_vs_b4_hierarchical_all_three": hierarchical_all,
            "b8_vs_b4_hierarchical_final_code_two": hierarchical_final,
            "b8_vs_b9_seed2911": compare_models(
                models["b9_seed2911"], models["b8_seed2911"], bootstrap_seed=2911
            ),
        },
    }
    artifact = root / "artifacts" / "selective_study_summary_v1.json"
    report = root / "SELECTIVE_STUDY_RESULTS.md"
    artifact.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report.write_text(render_markdown(summary))
    print(f"wrote {artifact}")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
