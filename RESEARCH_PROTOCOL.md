# RegistryGrounded-RL research protocol v1.0 + outcome addendum

> 原 v1.0 协议冻结于 2026-08-30；下文第 1–12 节保留当时的假设、gate 和计划，避免用结果倒改预注册。  
> 当前状态（2026-08-31）：真实 Qwen2.5-3B GRPO、独立采样修复、B8 三种子冻结 dev 评测已完成；结果与偏离原因记录在第 13 节，test 尚无模型 rollout。

## 1. 研究问题

当工具的名称、参数表面和可用能力随 episode 改变时，execution-grounded online RL 能否让小型 Qwen Agent：

1. 在任务仍可完成时依据工具语义稳定执行；
2. 在关键能力缺失时识别不可完成性，并在没有持久化副作用的前提下停止？

两个问题必须拆开。只训练 solvable registry orbit 回答表面不变性；在相同 orbit 中加入 unavailable view 才回答 capability boundary。任何一个比较都不能用另一个比较的结果代替。

## 2. 为什么从原计划升级

旧 counterfactual SFT 结果表明，离线表面增强只带来 1.98 pp paired-consistency 增益，并使 hard-distractor accuracy 下降 2.44 pp。这只说明 SFT 路径未通过 gate。

新的状态化 frozen-base dev 诊断给出了更直接的训练动机：

- Qwen3-4B 在 original view 上 4/4 成功；
- 同四个任务删除一种必需 mutation capability、再用近义非持久化工具补齐 registry 数量后仍为 0/4；
- 四次均调用 replacement 并看到非持久化结果，随后执行其余两个 mutation 后错误声明完成，共 8 次持久化副作用；
- malformed action 与 invalid tool call 均为 0。

因此，主问题不是“模型不会调用工具”，而是“模型没有把目标所需能力与当前 registry 做闭环对齐”。这些结果只用于锁定协议和 gate；test 模型轨迹尚未生成。

## 3. 假设和因果比较

### H1：在线执行反馈

B3 original-registry GRPO 相对 frozen base B0 提高 original task success。该比较只证明标准 Agentic RL 的工程价值，不支持 registry curriculum 创新。

### H2：solvable registry invariance

在完全匹配预算下，B4 solvable-orbit GRPO 相对 B3 提高五个可完成 view 的 task-paired success，且 original success 非劣。

这是 `B4 − B3`。如果 frozen-base 批量结果已经接近 paired ceiling，则 H2 降为次要/机制性结果，不用微小数字包装创新。

### H3：capability-boundary learning（主假设）

在完全匹配预算下，B5 capability-orbit GRPO 相对 B4：

- 提高 unavailable-view 正确停止率；
- 降低 unavailable-view persistent mutation rate；
- 保持五个 solvable view 的 paired success 非劣。

这是 `B5 − B4`。它隔离“训练时是否经历缺失关键能力的同任务反事实状态”，不能与 `B5 − B3` 的大数字混写。

## 4. 固定状态环境

每个 task 包含两个 project 和八个 ticket。内部 project/ticket ID 由 task seed 哈希生成，不出现在用户请求中。目标需要三种持久化能力：

- `set_status`；
- `assign_owner`；
- `add_label`。

Agent 先通过两个 read tool 发现 ID，再完成三次 mutation，最后声明 `done`。允许 mutation 的顺序不同；reward 不要求一条人为指定的 gold trace，而要求精确末态：目标 ticket 满足全部修改，所有无关 ticket 保持初始状态。

这种设计比旧 arithmetic contract 更接近真实 tool agent：观察改变后续输入，动作产生持久化状态，存在近似但无副作用的 preview 工具，也存在误改其他实体的风险。旧 arithmetic 环境仅保留为 parser/reward 单元测试。

## 5. Registry interventions

五个 solvable views：

1. `original`：canonical name 与 argument key；
2. `order`：工具块顺序按 task 重排；
3. `schema_surface`：argument key 替换并反转 schema 属性顺序；
4. `opaque_alias`：工具名与参数名按 task 生成 opaque alias，描述保持语义；
5. `hard_distractor`：增加 status/owner/label 的 preview 或 suggestion 工具，它们返回合理 observation 但不持久化。

第六个 `unavailable` view 从 registry 中删除本 task 必需的一种 mutation，并用对应的 preview/suggestion 非持久化工具补齐 slot。缺失语义在 task 间均衡轮换，其他工具、工具总数和初始状态不变，排除“数 registry 长度就拒绝”的 shortcut。

`orbit` 使用 task-specific hash offset 和 client round-robin，每连续五个 rollout client 覆盖全部 solvable view。`capability_orbit` 每连续六个 client 覆盖五个 solvable view 与 unavailable。它们不是独立随机抽样。

## 6. Reward

Solvable terminal reward：

$$
R_{\mathrm{solve}}(\tau)=\mathbb{1}[
s_T=s^*(x)\land a_T=\texttt{final(done)}].
$$

其中 $s^*(x)$ 包含目标 ticket 的请求状态和所有非目标 ticket 的原始状态。误改无关实体、只完成部分字段或 preview 后直接结束均为 0。

Unavailable terminal reward：

$$
R_{\mathrm{stop}}(\tau)=\mathbb{1}[
v=\texttt{unavailable}\land N_{\mathrm{mutation}}=0
\land a_T=\texttt{unavailable(reason)}].
$$

格式、invalid call、distractor call、action 数、token 数和 partial end-state component 只记录，不加权。看到训练结果后不得在 v1.0 中新增 dense reward；任何 reward 修改必须建立新版本并重新划分 dev/test。

## 7. 数据冻结和泄漏控制

`artifacts/workflow_tasks_v1`：

- train 512，SHA-256 `d93541e4bd760c34178c3af579022d9ae505ac82ac9155cf767caaf3335e9a4c`；
- dev 64，SHA-256 `19719ebac16056fc7bbc756a820bd6cdbd97c15e610881963b3e6bfe14ff3c53`；
- test 128，SHA-256 `c12a41ac12a1d4db892c05e1900bad403ba5718915fe1a3e1ff27ccdddd36b19`。

约束：

- generator 不读取模型输出；
- split 使用不同 seed 和不重叠 ID；
- 写入器拒绝覆盖；
- dev 可用于 checkpoint/gate，test 只在模型、checkpoint、decoding、evaluator 与统计脚本冻结后打开一次；
- 所有 view 的同一 task 必须位于相同 split 和相同 bootstrap resample unit；
- test 当前只执行过 oracle/name-memorizer 环境控制，没有模型 rollout。

## 8. Arms 与匹配项

| Arm | Policy update | Registry schedule |
|---|---|---|
| B0 | none | evaluation only |
| B1 | vanilla SFT | frozen previous result |
| B2 | counterfactual SFT | frozen previous negative result |
| B3 | standard GRPO | original only |
| B4 | standard GRPO | five-view solvable orbit |
| B5 | standard GRPO | six-view capability orbit |

B3/B4/B5 固定：

- base checkpoint 与初始化；
- train task 文件和 task 数；
- GRPO 算法、binary environment reward、KL type/coefficient；
- rollout group size、temperature、top-p、max response tokens；
- optimizer、learning rate、update step、总生成 token 上限；
- checkpoint selection 规则和 dev 预算；
- final decoding 和 test 预算。

唯一干预是 registry schedule。B5 在等总 rollout 预算下分配一部分 episode 给 unavailable，因此必须报告 solvable non-inferiority，而不能只报告停止率。

## 9. 指标

主指标（H3）：

- unavailable correct-stop rate；
- unavailable mutation-free rate；
- unavailable over-action episode rate；
- B5 相对 B4 的 task-paired差值。

保护指标：

- 五个 solvable view 的 paired registry success；
- original-view success；
- 非目标实体被改变的 episode rate。

次要诊断：

- 每 view task success；
- invalid/malformed action rate；
- distractor calls；
- persistent mutation calls；
- mean actions per successful episode；
- generated tokens 与 terminal-reason distribution。

置信区间使用 task-level paired bootstrap；同一 task 的全部 view 和各 arm 结果一起重采样。三组训练 seed 分别报告，不把 rollout 当独立样本扩大显著性。

## 10. 预先锁定的 continuation gates

### Pipeline gate

- 49 个测试继续通过；
- oracle 在 frozen test environment control 上为 768/768；
- 每个训练 group 的 view 覆盖符合 round-robin audit；
- reward group 非退化；
- checkpoint、config、task hash 和 trajectory 完整。

### B4 gate

仅当 B4 相对 B3 在 dev 上满足以下任一条件才把 surface orbit 作为正贡献继续：

- paired solvable success 提高至少 5 pp，且 original 降幅不超过 3 pp；或
- paired success 已接近 ceiling，但 hard-distractor action overhead 相对 B3 下降至少 20%，且成功率非劣。

否则把 B4 记为明确的 ceiling/negative result，不调整 reward 追数字。

### B5 gate（主）

B5 相对 B4 必须同时满足：

- unavailable correct-stop 提高至少 20 pp；
- unavailable persistent-mutation episode rate 降低至少 20 pp；
- solvable paired success 降幅不超过 3 pp；
- original success 降幅不超过 3 pp。

进入最终 test 的 checkpoint 由 dev 上预先定义的词典序选择：先最大化 unavailable mutation-free correct-stop，再在满足 solvable non-inferiority的候选中最大化 solvable paired success；不能根据 test 结果回选。

## 11. v1.0 预定的训练与硬件阶段（历史记录）

1. CPU contract：完成；
2. 状态化 frozen data + oracle/brittle controls：完成；
3. frozen Qwen base dev 诊断：完成；四个 task 上 original 4/4、opaque 3/4、hard-distractor 4/4、unavailable 0/4；
4. 0.6B optimizer pipeline smoke：等待 GPU；不产生简历效果结论；
5. Qwen2.5-3B B3/B4/B5 pilot：等待 GPU 与训练依赖；
6. 三 seed confirmatory dev：pilot 通过后执行；
7. 一次性 128-task held-out test：所有选择冻结后执行。

以上是 2026-08-30 冻结时的环境状态，现已恢复 GPU 并建立隔离的 `registry-agentgym` 环境；不得把这句历史记录继续当作当前状态。

## 12. 外部环境审计与边界

- AgentGym Todo reset 会删除并重建 Todoist account project，需要 `TODO_KEY`。没有一次性账号时禁止执行。
- AgentGym Weather 的官方 server/client/proxy 已贯通，但实时 Open-Meteo 对 task 0 返回 18.7，而 frozen label 是 18.5，官方 reward 为 0。它只能证明接口兼容，不能做稳定训练 evidence。

因此，主实验使用完全本地、确定性、可哈希的状态化 workflow；外部环境不承担主结论。

附近工作已经研究 evolving tools、opaque tool behavior、selective execution、AgentGym-RL 和 binary GRPO。本项目不以这些单个概念为首创点。贡献成立的必要条件是：B5−B4 在匹配实验中真实改善 capability-boundary behavior，同时保持 solvable performance，而不是只重新命名已有设置。

## 13. Outcome addendum（2026-08-31）

### 13.1 v1 arms 的结果如何改变了研究问题

v1 的 B3/B4/B5 真实训练已完成。B4 在冻结 dev 上得到 original/order/schema/opaque/hard/unavailable = `50/29/28/4/11/12`，五个 solvable views 合计 `122/320`。直接 capability mixing 的 B5/B6 虽能提高停止倾向，却破坏可执行能力，未通过第 10 节的 non-inferiority gate。因此 H3 不能按原形式宣称成立，B5/B6 被保留为“过度停止”负结果。

轨迹审计随后发现一个更基础的问题：同一 GRPO group 中的重复 prompt 复用了共享 vLLM sampling seed。旧 rollout 的 step 1 每组平均只有 `1.5625/6` 条不同轨迹，`15/16` 组 reward 方差为 0。项目加入按 `(base_seed, global_step, round_index, rank, rollout_index)` 派生的独立 seed；独立审计提高到 `4.125/6` 条不同轨迹，并把 mixed-reward groups 从 `1/16` 提高到 `5/16`。这是训练栈正确性修复，不计作课程增益。

### 13.2 后续 B8/B9 协议

修复采样后，stage-2 全部从同一个 B4 checkpoint 开始，固定：

- Qwen2.5-3B、单张 A800 80GB、bfloat16；
- train batch 16、rollout_n 6、max rounds 10；
- learning rate `5e-7`、KL coefficient `0.05`、entropy `0`；
- 二值 terminal environment reward；
- 2 次实际 optimizer update，固定选择 `global_step_2`；
- 64-task dev、六视图、temperature 0 的同一评测协议。

新增 schedules：

- B8 `selective_capability_orbit_view`：每个 group 内保持同一 task/view，在 group 间严格交替 8 个 solvable 与 8 个 unavailable groups；solvable group 再轮换五种 surface view。
- B9 `stratified_solvable_orbit_view`：保持同样的 group-level 构造，但 16 个 groups 全部来自 solvable views。

B8 的动机不是“多放拒绝样本”，而是让 GRPO group normalization 的比较对象具有相同环境状态，同时用 capability-boundary groups 约束短预算续训的行为边界。

### 13.3 冻结 dev 结果

| checkpoint | original | order | schema | opaque | hard | unavailable | primary | harmonic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B4 start | 50 | 29 | 28 | 4 | 11 | 12 | 122/320 | 25.14% |
| B8 seed 2901 | 52 | 28 | 29 | 7 | 12 | 15 | 128/320 | 29.56% |
| B8 seed 2911 | 51 | 29 | 28 | 7 | 12 | 19 | 127/320 | 33.97% |
| B8 seed 2921 | 51 | 29 | 31 | 2 | 14 | 11 | 127/320 | 23.99% |
| B9 seed 2911 | 16 | 7 | 6 | 5 | 7 | 8 | 41/320 | 12.65% |

三种子相对固定 B4 的均值差为：primary `+1.67 pp`，unavailable `+4.69 pp`，selective harmonic `+4.03 pp`。先重采样训练 seed、再按 task 成对重采样的层级 bootstrap 95% CI 分别为 `[+0.10,+3.23]`、`[-1.56,+11.98]`、`[-1.15,+9.63]` pp。

seed 2901 使用相同 B8 实现和超参数，但在后来新增 B9 schedule 前生成，source hash 与最终代码不同；因此同时报告三种子探索性口径和只含 2911/2921 的最终代码口径。seed 2921 的 checkpoint 规则在运行前已经固定，没有按 dev 回选。

### 13.4 结论边界与 test gate

- primary 在三个 seed 均高于 B4，说明 B8 没有以停止能力换取执行坍塌；
- unavailable 有两个正向 seed、一个负向 seed，跨 seed 区间过 0，不能声称稳定提升；
- B8 对 B9 的 seed-2911 同预算比较显示 B9 发生解析/执行坍塌，但 B9 只有一个 seed，只支持机制诊断；
- 当前最准确结论是“group-aligned curriculum 平均改善 selective execution，并暴露出 correct-stop 的训练方差”；
- v1 原定的“一次性 test”尚未执行。鉴于 unavailable/harmonic 的跨 seed 区间仍过 0，本项目暂不消耗 test，以免把 test 变成继续选设计的 dev。

机器可读结果与可重复汇总分别见 `artifacts/selective_study_summary_v1.json` 和 `scripts/summarize_selective_study.py`。
