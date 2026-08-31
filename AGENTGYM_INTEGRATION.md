# AgentGym-RL integration

## 当前状态与固定版本

集成已在真实 A800 上完成训练、保存、导出和冻结评测，不再是仅配置好的待运行路径。

- AgentGym-RL commit：`82402a99c62a293735a3f412fb8ac9a600673bc0`；
- AgentGym submodule：`d014732d9fe39b975c368c03749bfd50950067f6`；
- 最终项目代码 aggregate hash：`e8faf131647e7b544a2f601993aa0335863d2adfcfa2c396898d45a600297e18`；
- rollout patch hash：`0158d7573957aa2af0f64ac6184492be0f9945114909d5d30e71529cefa56b31`；
- Python 环境：独立的 `registry-agentgym` 环境；
- PyTorch 2.4.0+cu124、Transformers 4.51.3、Qwen2.5-3B-Instruct、bfloat16。

本地 client patch 注册 `registrygrounded`、`registryworkflow`、`registrytodo` 和 `registryweather`。主环境 client 实现官方 rollout 接口：

```text
reset(item_id)
observe()
step(model_text) -> StepOutput(state, reward, done)
```

环境 terminal scalar reward 只落到最后一个 response token；项目没有 learned reward model。

## RegistryWorkflow 地址与 schedules

```text
workflow:///absolute/path/train.jsonl?view=original&seed=1701&group_size=6
workflow:///absolute/path/train.jsonl?view=orbit&seed=1701&group_size=6
workflow:///absolute/path/train.jsonl?view=capability_orbit&seed=1701&group_size=6
workflow:///absolute/path/train.jsonl?view=selective_capability_orbit&seed=2911&group_size=6
workflow:///absolute/path/train.jsonl?view=stratified_solvable_orbit&seed=2911&group_size=6
```

| view schedule | Arm | 组内/组间语义 |
|---|---|---|
| `original` | B3 | original only |
| `orbit` | B4 | 旧 client-level 五种 solvable orbit |
| `capability_orbit` | B5 | 旧 client-level 五种 solvable + unavailable；负结果 |
| `selective_capability_orbit` | B8 | 组内同 task/view；组间 8 solvable + 8 unavailable |
| `stratified_solvable_orbit` | B9 | 组内同 task/view；组间 16 solvable |

B8/B9 通过 `group_size=rollout_n` 对齐 AgentGym client serial 与 GRPO group：六条候选必须看到完全相同的初始状态和 registry，不能把不同 view 混入一次 group-relative normalization。

## 关键正确性修复：独立 rollout seed

pinned AgentGym-RL 原路径把一个 `SamplingParams.seed` 复用于重复 prompt。同一 GRPO group 的六个候选因此经常生成完全相同的动作序列；旧审计为：

- mean unique trajectories/group：`1.5625/6`；
- fully duplicated groups：`8/16`；
- mixed-reward groups：`1/16`；
- zero-reward-variance groups：`15/16`。

修复由 `registry_grounded_rl/rollout_seeding.py` 生成确定性的独立 seed：

```text
seed = H(base_seed, global_step, round_index, rank, rollout_index)
```

vendor rollout 对每个未结束 trajectory 深拷贝 `SamplingParams` 并写入对应 seed。首次独立采样审计为 `4.125/6`、`1/16`、`5/16`、`11/16`。独立 seed 保证随机流不同，不保证策略最终行为一定不同；因此每个正式 run 仍必须审计 unique trajectory 与 reward variance。

## 训练与评测入口

统一训练入口：

```bash
cd /path/to/registry-grounded-rl

CUDA_VISIBLE_DEVICES=1 \
AGENTGYM_ROOT=/path/to/AgentGym-RL \
ARM=selective_capability_orbit \
RUN_SEED=2911 \
MODEL_PATH=/path/to/b4/huggingface \
N_GPUS=1 TRAIN_BATCH_SIZE=16 ROLLOUT_N=6 \
ACTOR_LR=5e-7 KL_LOSS_COEF=0.05 ENTROPY_COEFF=0 \
TOTAL_TRAINING_STEPS=3 \
  bash scripts/train_agentgym_grpo.sh
```

`TOTAL_TRAINING_STEPS=3` 在当前 trainer 中产生 step 1/2 两次实际 optimizer update；报告固定使用 `global_step_2`。导出与审计：

```bash
python \
  scripts/audit_rollout_groups.py /path/to/run \
  --task-file artifacts/workflow_tasks_v1/train.jsonl --rollout-n 6

python \
  scripts/export_single_gpu_hf_checkpoint.py \
  --actor-dir /path/to/run/checkpoints/global_step_2/actor
```

冻结评测固定 dev task hash、temperature 0、top-p 1、max rounds 10：

```bash
CUDA_VISIBLE_DEVICES=1 \
AGENTGYM_ROOT=/path/to/AgentGym-RL \
MODEL_PATH=/path/to/global_step_2/actor/huggingface \
MODEL_TAG=my_checkpoint \
OUTPUT_ROOT=/path/to/eval_root \
RUN_SEED=1701 BATCH_SIZE=24 GPU_MEMORY_UTILIZATION=0.55 \
  bash scripts/run_frozen_eval_matrix.sh
```

最终汇总会校验六个 view 的 task/data/decoding/model hash：

```bash
python \
  scripts/summarize_selective_study.py
```

## 每个 run 的 claim gate

1. `run_manifest.json` 中保存 base 权重、task/index、代码、vendor rollout 和运行时 hash；
2. 六条候选组成完整同 item、同 view group；
3. 独立 seed strategy 为 `independent-per-trajectory-round-v1`；
4. 至少一个 group 有 mixed reward，否则该步没有相对 advantage 信号；
5. train log 有有限的 grad norm、KL、reward 和两次 optimizer update；
6. checkpoint 能导出为 Hugging Face safetensors；
7. 每个冻结 view 恰有 64 个不重复 item，并保存 trajectory/eval log hash；
8. 多 seed 结论以 seed 为外层统计单位，不把 rollout 或 task 冒充独立训练复现。

## 外部 AgentGym 审计

- Todo client 会在 reset 删除并重建真实账号 project/task；没有一次性账号时禁止运行。
- Weather server/client/proxy 已贯通，但 live Open-Meteo 返回值与 frozen label 漂移，terminal reward 因此为 0。

二者只证明接口兼容和外部环境风险；主训练使用进程内、确定性、可哈希的 RegistryWorkflow。
