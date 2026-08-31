# RegistryGrounded-RL：独立采样与选择性执行研究结果

生成时间：2026-08-31T08:02:16.245814+00:00  
状态：stage-2 冻结 dev 研究已完成；test split 未做模型 rollout，仍保持未触碰。

## 结论先行

1. **先修复训练信号，再讨论课程。** 原 AgentGym-RL 路径给同一 GRPO group 的重复 prompt 复用了同一个采样 seed，旧审计每组平均只有 1.56 条独立轨迹、15/16 组零 reward 方差。项目改成按 `(base seed, global step, round, rank, rollout index)` 派生采样 seed；独立采样审计每组平均提高到 4.12 条，恢复了可学习 advantage。
2. **简单有效的干预是 group-aligned selective curriculum（B8）。** 每个 GRPO group 内保持同一 task 和同一 registry 状态，group 间固定 50:50 轮换可执行与不可执行状态。这样 advantage 比较的是同一环境中的动作质量，不把不同 registry 难度混进组内归一化。
3. **三训练种子相对固定 B4 起点：**五个可执行视图均值由 38.12% 到 39.79%（+1.67 pp）；unavailable 由 18.75% 到 23.44%（+4.69 pp）；执行—停止 harmonic mean 由 25.14% 到 29.17%（+4.03 pp）。
4. **结论有方向性，但不能包装成“稳定显著”。** 第三个 seed 的 unavailable 为 11/64，略低于 B4 的 12/64；三种子层级 bootstrap 的区间见下表。当前可信表述是“平均改善并定位出种子方差”，不是“所有种子都提升”。
5. **同 seed 机制对照支持 capability-boundary exposure 的作用。** seed 2911 下，B8 相比只继续训练 solvable views 的 B9，可执行成功率高 +26.88 pp（95% task bootstrap [+21.87 pp, +32.19 pp]），harmonic mean 高 +21.31 pp（[+12.12 pp, +30.77 pp]）。B9 的下降伴随大量解析错误，因此 unavailable group 在这里同时是行为边界和短预算稳定器，不能简化成“多加拒绝样本”。

## 冻结评测主表

每个 cell 是同一组 64 个 dev tasks、temperature=0、最多 10 轮。Primary 是前五个可执行视图；harmonic mean 同时惩罚“只会执行”和“只会停止”。

| checkpoint | original | order | schema | opaque | hard | unavailable | primary | harmonic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| b4_start | 50/64 | 29/64 | 28/64 | 4/64 | 11/64 | 12/64 | 122/320 (38.12%) | 25.14% |
| b8_seed2901 | 52/64 | 28/64 | 29/64 | 7/64 | 12/64 | 15/64 | 128/320 (40.00%) | 29.56% |
| b8_seed2911 | 51/64 | 29/64 | 28/64 | 7/64 | 12/64 | 19/64 | 127/320 (39.69%) | 33.97% |
| b8_seed2921 | 51/64 | 29/64 | 31/64 | 2/64 | 14/64 | 11/64 | 127/320 (39.69%) | 23.99% |
| b9_seed2911 | 16/64 | 7/64 | 6/64 | 5/64 | 7/64 | 8/64 | 41/320 (12.81%) | 12.65% |

## B8 相对 B4：逐训练种子

| seed | primary 差值及 task-bootstrap 95% CI | unavailable 差值 | harmonic 差值及 95% CI |
|---:|---:|---:|---:|
| 2901 | +1.88 pp [-0.63 pp, +4.38 pp] | +4.69 pp | +4.42 pp [-1.11 pp, +10.34 pp] |
| 2911 | +1.56 pp [-1.25 pp, +4.38 pp] | +10.94 pp | +8.83 pp [+3.51 pp, +15.29 pp] |
| 2921 | +1.56 pp [-0.94 pp, +4.06 pp] | -1.56 pp | -1.15 pp [-4.73 pp, +0.82 pp] |

seed 2901 是探索性复现：B8 实现与超参数相同，但随后为加入 B9 调度产生了加法式 source hash 变化。seed 2911/2921 使用完全相同的最终代码。checkpoint 选择规则在 seed 2921 运行前已固定为第二次 optimizer update，不按 dev 挑选。

## 跨训练种子的层级 bootstrap

先重采样训练 seed，再在所选 seed 内按 task 成对重采样；这避免把 64 个 task 当作 64 次独立训练。左侧 B4 是固定 stage-1 checkpoint，因此区间衡量的是 stage-2 B8 续训的随机性，不是假装 B4 也有三个训练复现。

| 范围 | primary | unavailable | harmonic |
|---|---:|---:|---:|
| 全部三 seed（含探索性 2901） | +1.67 pp [+0.10 pp, +3.23 pp], P(>0)=0.981 | +4.69 pp [-1.56 pp, +11.98 pp], P(>0)=0.899 | +4.03 pp [-1.15 pp, +9.63 pp], P(>0)=0.927 |
| 最终代码两 seed（2911/2921） | +1.56 pp [-0.31 pp, +3.44 pp], P(>0)=0.939 | +4.69 pp [-3.12 pp, +14.84 pp], P(>0)=0.743 | +3.84 pp [-2.56 pp, +11.61 pp], P(>0)=0.782 |

## 训练确实发生且 GRPO 信号非退化

| run | optimizer updates | rollout group 审计 |
|---|---:|---|
| b8_seed2901 | 2 | step1: unique=3.56, mixed=5/16; step2: unique=3.81, mixed=6/16 |
| b8_seed2911 | 2 | step1: unique=4.00, mixed=6/16; step2: unique=3.25, mixed=4/16 |
| b8_seed2921 | 2 | step1: unique=3.19, mixed=4/16; step2: unique=2.56, mixed=5/16 |
| b9_seed2911 | 2 | step1: unique=2.88, mixed=2/16; step2: unique=3.31, mixed=1/16 |

四个 stage-2 run 均从同一 B4 权重 `0e527799a9713d5afc7b3e149a660fb80b8691ffc1e06341615a9b19a49e033d` 开始，使用 Qwen2.5-3B、单张 A800 80GB、batch 16、每 task 6 条 rollout、二值精确末态 reward、学习率 5e-7、KL 0.05、entropy 0，共 2 次实际 optimizer update。seed 2911 的 B8/B9 除 registry schedule 外，代码、起点、数据、算法、预算与随机种子均相同。

## 研究边界

- 这是冻结 **dev** 上的短预算 stage-2 研究，不是外部 benchmark SOTA，也不是 held-out test 结论。
- B8 三 seed 的 primary 总体稳定，但 opaque/schema 的能力分配存在波动；unavailable 也有一个负向 seed。
- B9 是一个 seed 的机制性对照，足以暴露失败模式，但不足以估计 B9 的跨 seed 均值。
- 不声称首创 selective execution、dynamic tools 或 GRPO。项目贡献是把 registry intervention、可执行 verifier、GRPO 分组语义和 rollout RNG 审计连成一条可证伪实验链。

## 当前可用于简历的一句话

> 针对工具 Agent 在 registry 变化或关键能力缺失时盲目执行的问题，我在 Qwen2.5-3B/AgentGym-RL 上构建可验证多轮环境，修复同组 rollout 复用随机种子导致的 GRPO 信号退化，并提出组内同状态、组间 50:50 能力边界课程；三训练种子冻结 dev 上在保持可执行成功率的同时，将正确停止率平均提高 +4.69 pp，但保留一个负向种子并用层级 bootstrap 报告不确定性。

复现汇总：

```bash
cd /path/to/registry-grounded-rl
python scripts/summarize_selective_study.py
```

机器可读证据：`artifacts/selective_study_summary_v1.json`。
