# RegistryGrounded-RL

一个面向多轮工具 Agent 的可验证后训练项目：研究模型能否依据当前 registry **实际提供的能力**行动——可完成时跨工具表面执行，关键能力缺失时零副作用停止。

项目已完成 Qwen2.5-3B 的真实 AgentGym-RL/verl/vLLM GRPO、checkpoint 导出、rollout 审计和 64 tasks × 6 views 的冻结 dev 评测。最终结果见 [`SELECTIVE_STUDY_RESULTS.md`](SELECTIVE_STUDY_RESULTS.md)；128-task test 尚未进行模型 rollout。

## 公开仓库边界

本仓库发布项目自有的动态工具环境、GRPO 接口、独立 rollout seed 修复、训练与评测脚本、72 项测试、冻结任务和汇总结果；不发布 Qwen 权重、FSDP/Hugging Face checkpoint、完整 trajectory、训练日志或 vendored AgentGym-RL/verl。

对 AgentGym-RL 的两处最小修改以
[`patches/agentgym-rl-registry-grounded.patch`](patches/agentgym-rl-registry-grounded.patch)
提供，并在 `AGENTGYM_INTEGRATION.md` 中固定上游 revision。结果数字可由公开结果表和紧凑指标工件审查，但完整训练复现仍需要本地模型、上游训练栈及未发布的大体积运行工件。

## 快速检查

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
registry-grounded-rl --help
```

## 结论先行

- 原 AgentGym-RL rollout 路径对同组重复 prompt 复用采样 seed，旧审计每组平均只有 1.56 条独立轨迹，15/16 组没有 reward 方差；本项目实现逐 trajectory/round 的确定性独立 seed，使独立审计均值提高到 4.12 条并恢复 GRPO advantage 信号。
- 朴素地把可执行与 unavailable 状态混进训练会学出“立即停止”捷径；只继续训练 solvable views 又在短预算下发生执行/格式坍塌。
- 最终 B8 课程保持每个 GRPO group 内的 task 与 registry 状态相同，在 group 间按 50:50 轮换 solvable/unavailable，使组内 advantage 真正比较同一环境中的动作质量。
- B8 三训练种子相对固定 B4 起点，五个可执行视图由 38.12% 提高到 39.79%（+1.67 pp），unavailable correct-stop 由 18.75% 平均提高到 23.44%（+4.69 pp），执行—停止 harmonic mean 由 25.14% 到 29.17%（+4.03 pp）。
- unavailable 与 harmonic 的跨 seed 95% 区间仍跨 0，且 seed 2921 的 unavailable 为 11/64、略低于 B4 的 12/64。因此可以说“平均改善、执行能力稳定并暴露出停止方差”，不能说“所有种子稳定显著提升”。

## 为什么做

真实 Agent 的工具名称、参数 schema、候选集合和可用能力会随会话与部署环境变化。模型即使会基本 tool use，也可能：

1. 记住工具名或参数表面，而没有依据描述和执行结果选工具；
2. 在关键能力缺失时继续执行剩余步骤、产生副作用，最后错误声称完成。

旧 `RegistryShift-Align` 的 counterfactual SFT 只比 token-matched vanilla SFT 提高 1.98 pp paired consistency，并使 hard-distractor accuracy 下降 2.44 pp。这只是“离线模仿不足”的先验证据；本项目的根本问题是：**执行反馈能否让 Agent 对齐目标所需能力与当前环境实际能力？**

## 可执行状态环境

每个任务有两个 project、八个 ticket 和按 task 随机生成的隐藏 ID。Agent 需要先用 read tools 找到目标，再持久化修改 status、assignee 和 label。六个 registry views 为：

- `original`：标准名称与参数；
- `order`：工具顺序置换；
- `schema_surface`：参数 key 和 schema 属性顺序改变；
- `opaque_alias`：episode-local opaque 工具名与参数名，描述保留语义；
- `hard_distractor`：加入相似但不持久化的 preview/suggestion；
- `unavailable`：删除本任务必需的一种 mutation capability，并用近义非持久化工具补齐 slot，工具总数不变。

可执行任务的二值 reward 为：

$$
R_{\mathrm{solve}}=\mathbb{1}[
\text{目标末态精确正确}\land
\text{无关记录不变}\land
\text{最终声明 done}].
$$

能力缺失任务的 reward 为：

$$
R_{\mathrm{stop}}=\mathbb{1}[
\text{必需能力缺失}\land
\text{零持久化副作用}\land
\text{最终声明 unavailable}].
$$

格式、token 数、部分完成和 action 数只作诊断，不混入 dense reward。

## 实验如何自然演化

| Arm | 起点/训练 schedule | 结果与作用 |
|---|---|---|
| B3 | base → original-only GRPO | 证明在线执行 reward 有工程价值 |
| B4 | base → 五种 solvable orbit GRPO | 作为最终 stage-2 固定起点；dev primary 122/320 |
| B5/B6 | base → 混入 capability 状态 | unavailable 提高但执行能力坍塌，定位“过度停止” |
| independent B7 | B4 → solvable orbit，修复采样 seed | 恢复组内多样性，但第二步开始退化 |
| B8 | B4 → group 间 50:50 solvable/unavailable | 最终方法；组内同状态、三种子复现 |
| B9 | B4 → group 间仅 solvable，seed 2911 | 与 B8 同 seed/同代码/同预算的机制对照；发生解析与执行坍塌 |

B8 不靠复杂 loss。它修正的是 GRPO 的比较单位：若同组候选处于不同 registry 状态，group normalization 会把“环境难度”误当成“动作好坏”；同组保持环境一致后，relative advantage 才有清楚语义。

## 冻结 dev 主结果

| checkpoint | original | order | schema | opaque | hard | unavailable | primary | harmonic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B4 start | 50/64 | 29/64 | 28/64 | 4/64 | 11/64 | 12/64 | 122/320 | 25.14% |
| B8 seed 2901 | 52/64 | 28/64 | 29/64 | 7/64 | 12/64 | 15/64 | 128/320 | 29.56% |
| B8 seed 2911 | 51/64 | 29/64 | 28/64 | 7/64 | 12/64 | 19/64 | 127/320 | 33.97% |
| B8 seed 2921 | 51/64 | 29/64 | 31/64 | 2/64 | 14/64 | 11/64 | 127/320 | 23.99% |
| B9 seed 2911 | 16/64 | 7/64 | 6/64 | 5/64 | 7/64 | 8/64 | 41/320 | 12.65% |

三种子层级 bootstrap 先重采样训练 seed，再按 task 成对重采样：primary 差值 +1.67 pp，95% CI `[+0.10, +3.23]`；unavailable +4.69 pp，`[-1.56, +11.98]`；harmonic +4.03 pp，`[-1.15, +9.63]`。seed 2901 是 B8 实现相同但 source hash 较早的探索性复现；最终代码严格复现为 2911/2921，报告中两种口径均单独列出。

## 训练配置与证据

- 模型：Qwen2.5-3B-Instruct；stage-2 统一从 B4 权重 SHA-256 `0e527799a9713d5afc7b3e149a660fb80b8691ffc1e06341615a9b19a49e033d` 开始；
- 硬件：单张 NVIDIA A800-SXM4-80GB；
- 栈：AgentGym-RL + verl + vLLM，PyTorch 2.4.0/CUDA 12.4，bfloat16；
- 配置：train batch 16、rollout_n 6、最多 10 轮、学习率 5e-7、KL 0.05、entropy 0、2 次实际 optimizer update；
- 冻结评测：64 dev tasks × 6 views、temperature 0、top-p 1、最多 10 轮；
- 每个 run 保存 manifest、源码/数据/起点 hash、训练日志、完整 trajectories、FSDP checkpoint 与导出的 Hugging Face 权重；
- 72 个单元/接口测试在最终文档更新前通过；最终交付再次运行全套测试。

关键入口：

- `registry_grounded_rl/workflow_environment.py`：持久化状态环境与末态 reward；
- `registry_grounded_rl/workflow_agentgym.py`：registry schedule；
- `registry_grounded_rl/rollout_seeding.py`：独立 rollout seed；
- `scripts/train_agentgym_grpo.sh`：统一训练入口；
- `scripts/audit_rollout_groups.py`：组内多样性/reward 方差审计；
- `scripts/summarize_selective_study.py`：从原始证据重算最终表与统计。

复现汇总：

```bash
python scripts/summarize_selective_study.py
```

上面的汇总脚本需要原始运行目录；公开仓库中的
`artifacts/release_metrics.json` 是从冻结工件导出的紧凑结果快照。

## License

项目自有代码采用 MIT License。AgentGym-RL/verl 的许可与补丁边界见
`NOTICE`。

## 数据与边界

| Split | Rows | SHA-256 |
|---|---:|---|
| train | 512 | `d93541e4bd760c34178c3af579022d9ae505ac82ac9155cf767caaf3335e9a4c` |
| dev | 64 | `19719ebac16056fc7bbc756a820bd6cdbd97c15e610881963b3e6bfe14ff3c53` |
| test | 128 | `c12a41ac12a1d4db892c05e1900bad403ba5718915fe1a3e1ff27ccdddd36b19` |

test 只运行过模型无关的 oracle/name-memorizer 环境控制，没有模型 rollout。当前不声称外部 benchmark SOTA、held-out test 泛化或概念首创。AgentGym Todo 会修改真实账号数据，Weather 的 live API 标签已经漂移，因此两者只作接口审计，主结论全部来自本地确定性、可哈希环境。

## 简历表述

> 针对工具 Agent 在 registry 变化或关键能力缺失时盲目执行的问题，我在 Qwen2.5-3B/AgentGym-RL 上构建可验证多轮环境，修复同组 rollout 复用随机种子导致的 GRPO 信号退化，并提出组内同状态、组间 50:50 能力边界课程；三训练种子冻结 dev 上在保持可执行成功率的同时，将正确停止率平均提高 4.69 pp，并用层级 bootstrap 如实报告一个负向种子与跨种子不确定性。
