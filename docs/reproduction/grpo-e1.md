# GRPO E1：失败样本学习与执行协议一致性

> 历史200题实验记录；本页选模与“主结果”仅适用于该历史系列。当前项目结论见[核心结果](../results.md)。

E1 从与上一版 GRPO 相同的 SFT 合并权重开始，已完成 500 次参数更新、验证选点、
模型导出及 Final-200 评测。上一版的 [实现与结果](grpo.md) 保留；
本说明描述 E1 实现，[结果与分析](grpo-e1-results.md) 单独记录本轮结果。

## 修改依据

上一版生成记录中，227 条提前结束、145 条连续非法动作轨迹被当作无效采样。
456 个组因包含无效轨迹而被过滤，其中也包含真正的 Reward 不可验证样本。
模型错误导致整个有奖励差异的组被丢弃，确实损失了可用于学习的信号。

## 奖励与过滤

| 情况 | 训练奖励 | 是否参与动态采样 |
|---|---|---|
| 环境正常终局，包括错误购买、重复循环、弃购、步数耗尽 | 原始 Reward v3 | 是 |
| 模型提前结束、连续 3 次守卫拒绝、耗尽响应或上下文预算，且环境未正常终局 | `-0.65` | 是 |
| 环境服务、观测协议或投影异常、Reward 不可验证 | 无学习信号 | 过滤所在组 |

`-0.65` 为单独的可配置训练成本，与 Reward v3 的循环失败惩罚同量级，
不是伪造的环境终局分数。它由 `configs/agent_loop.yaml` 的
`model_failure_reward` 指定。模型失败不产生严格成功，也不补造 `gold_purchase`。

AgentLoop 输出 `training_reward`；组内差异筛选和 GRPO 优势计算使用同一份实际
序列奖励。训练器检查它与轨迹声明的 `training_reward` 一致。
恒定奖励组仍被过滤，因为 GRPO 的组内相对优势为零。
本次不启用额外长度 shaping，也不改变学习率、采样温度或策略损失。

验证显式传递 `shopping_validate`，使用原始 Reward v3 的 `terminal_utility`；
未正常终局的模型失败仍为原评测口径的 0 分。新增训练成本不参与验证选点。

## 执行协议

训练和正式评测共享动作参数与页面守卫、非法动作反馈、连续拒绝上限和
多工具串行策略：每轮只执行第一个工具，其他调用记录为丢弃。
非法调用不执行环境动作，不消耗合法工具步数，并反馈当前可用商品、按钮和返回工具。
提前 final、连续 3 次非法调用及 35 个执行工具步的终止行为一致。
`think` 也计入执行工具步数。关闭 veRL 额外的 assistant/user 回合上限，避免它在
执行最后一个合法工具前抢先终止；动作步数、连续拒绝和 token 预算负责终止。
完整序列与响应预算仍为 24576/20480 token。
模型输入生成的 token 预算约束仍存在，不把预算耗尽误归因于基础设施故障。

## 独立训练与验证

使用相同的 1000 条训练任务、50 条独立验证任务、SFT 权重、随机种子及双 H100 配方。
从第 0 步开始，完成 500 次实际参数更新，每 10 步保存，每 50 步验证。
按独立验证平均原始奖励选点，同分取较早步数；Final-200 不参与训练和选点。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export NCCL_CUMEM_ENABLE=0
# SFT_MODEL 与 E1_OUTPUT 分别指定可访问的 SFT 合并模型和新输出目录。
python scripts/apply_verl_dynamic_sampling_patch.py
python scripts/train_grpo.py \
  --model "$SFT_MODEL" --output "$E1_OUTPUT" \
  --logger tensorboard --experiment-name shopping-grpo-e1 \
  -- +ray_kwargs.ray_init._node_ip_address=127.0.0.1 trainer.save_freq=10
```

新启动使用空输出目录，不从上一版 GRPO 检查点恢复。本次在检查点 10 完成保存和
HDFS 归档后主动暂停，将 E1 启动时的原生回合上限 105 改为 `null`，然后从完整
检查点 10 恢复。恢复包含模型、优化器、调度器、随机状态与数据迭代状态：

```bash
python scripts/train_grpo.py \
  --model "$SFT_MODEL" --output "$E1_OUTPUT" \
  --logger tensorboard --experiment-name shopping-grpo-e1 \
  --resume-from "$E1_OUTPUT/global_step_10" \
  -- +ray_kwargs.ray_init._node_ip_address=127.0.0.1 \
  trainer.save_freq=10 trainer.val_before_train=false
```

暂停前尚未保存的第 11 次更新及后续采样另存并舍弃，不混入正式诊断；最终正式
更新记录为连续的 1–500。此前训练诊断中的已保存轨迹最多 61 个原生回合，未触及 105 上限，
但这仍属于一次配置修正后的恢复运行，应在后续复现比较中保留这一事实。

原生 TensorBoard 区分训练奖励、
原始验证奖励、生成阶段模型/基础设施失败及采样组计数。
`generation_batch` 诊断保存各类训练奖励统计和接受的组；`optimizer_step` 保存
实际更新组数及接受轨迹的模型失败比例，跳过更新时 `group/trained=0`。
恢复使用 `purge_step=11` 和原生 SessionLog START 标记清除废弃的第 11 步标量，
保留原始事件时间和正式恢复后的记录。看板读取原生事件文件，不由诊断 JSON 补造曲线。

```bash
tensorboard --logdir "$E1_OUTPUT/tensorboard" \
  --host 127.0.0.1 --port 6006 --reload_interval 15
```

实际看板同时保留上一版训练，E1 的运行名为 `E1/.`。生成统计在同一次参数更新前
可能出现多个同 Step 点；以 `training/global_step` 和 `training/optimizer_updated`
判断更新进度，以 `val-core/shopsimulator/reward/mean@1` 判断独立验证表现。

## 代码与测试入口

| 范围 | 入口 | E1 变更 |
|---|---|---|
| 动作协议 | [actions.py](../../src/shopping_grpo/environment/actions.py)、[rollout.py](../../src/shopping_grpo/evaluation/rollout.py)、[tools.py](../../src/shopping_grpo/training/grpo/adapter/tools.py) | 共享参数守卫、反馈、连续拒绝上限与多调用串行规则 |
| 失败分类与奖励 | [runtime.py](../../src/shopping_grpo/training/grpo/adapter/runtime.py)、[agent_loop.py](../../src/shopping_grpo/training/grpo/adapter/agent_loop.py) | 模型失败保留、基础设施问题过滤，训练成本与环境分数分离 |
| 筛选与诊断 | [dynamic_sampling.py](../../src/shopping_grpo/training/grpo/dynamic_sampling.py)、[veRL 补丁](../../patches/verl-0.8.0-shopping-dynamic-sampling.patch) | 使用实际训练奖励筛组，核对优势输入，记录生成/更新/跳过 |
| 运行配置 | [grpo.yaml](../../configs/grpo.yaml)、[agent_loop.yaml](../../configs/agent_loop.yaml) | 关闭额外回合上限，配置模型失败训练成本 |
| 协议回归 | [test_grpo_protocol.py](../../tests/test_grpo_protocol.py) | 固定动作序列通过实际 veRL 工具处理与评测入口对照 |

veRL 仍锁定 0.8.0；补丁标记更新为 V4，安装器与预检同步校验该版本。
最终运行补丁和仓库源码哈希保存在 [结果数据](grpo-e1-results.json)。

217 项 Worker 回归中 216 项通过、1 项跳过。新增固定动作测试通过真实 veRL 工具处理
和正式评测入口，覆盖非法操作恢复、连续拒绝、缺失参数、损坏 JSON、未知工具、
多工具调用、提前结束和 `think` 步数上限；另验证原生回合计数不抢先终止合法工具、
响应预算仍生效、训练/验证奖励分离与采样依据一致。

本次运行证据单独保存于 `outputs/reproduction/grpo-e1-20260930/`。
`status.json` 为状态入口，`train.log` 为原始输出，`training_diagnostics.jsonl`
为采样及更新诊断，`run_manifest.json` 保存源码、数据和初始模型哈希。
后台监督、HDFS 归档及结束/失败后双卡保活沿用已验证的执行机制。
训练期低占空比保活独立于训练计算，不能作为训练吞吐或 SM 效率提升的证据。

完整结果、统计边界、归档位置和后续改进依据见 [结果与分析](grpo-e1-results.md)。
