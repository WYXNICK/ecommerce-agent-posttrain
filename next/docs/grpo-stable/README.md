# 稳定终局 GRPO（R-stable）

**正式训练运行中。** 北京时间2026-11-07 17:20在worker 1163758的双H100启动；[离线验收](../../results/grpo/stable/validation.json)、[启动验收与归档](../../results/grpo/stable/execution.json)。实时状态以`outputs/grpo/stable/progress.json`和`status.json`为准，原生TensorBoard 6016看板的`R-stable`分组展示曲线；完成后本目录`results.md`单独汇总成绩。

## 1. 为什么优先做这一版

- S-repair 起点 dev **226/300**；终局 GRPO 最佳 **228/300**，相对起点新增成功10题、丢失8题。局部信用最高 **225/300**，最终221题，未胜过起点。
- 局部信用非零增量只覆盖约 **0.25% 的训练动作 token**，还增加了分支估值成本。当前更值得验证的是：**扩大每次更新的任务覆盖、约束策略偏移，能否减少学会新任务时丢失原有能力。**
- 之前没有计算相对固定 SFT 的 KL，也没有完整分布熵；已有采样器/actor 概率差异不能证明策略未漂移。小批次和漂移是待检验假设，不是已经确诊的唯一原因。

证据见[终局结果](../grpo/results.md)、[局部信用结果](../grpo-local/results.md)。本版不同时修改奖励、增加局部信用、熵奖励或教师调用，避免收益来源更加难以解释。

## 2. 冻结配方

从同一个 **S-repair step72** 独立初始化 RL LoRA；原始 SFT、训练池、环境和评测结果均复用。

| 项目 | 本版设置 |
|---|---|
| 学生与数据 | Qwen3.5-2B；冻结 GRPO 池2,000题，dev300选模；测试不参与训练和调参 |
| 有效批次 | **8个不同任务 × 每题4条 = 32条轨迹/更新**；同批使用同一冻结策略 |
| 采样并发 | 每波2题×4条，最多8路；至少4波补齐有效批次；并发不随有效批次扩大 |
| 动态采样 | 原 training_reward 判断组内差异；最多32波，即64个候选组；不足8个有效组则整批不更新并记录，不用残缺批次凑更新 |
| 奖励与优势 | 可验证终局用 Reward v3；模型非终局失败−0.65；基础设施/不可验证过滤整组；`A=R−同题均值`，不除标准差 |
| KL | 固定 S-repair 为参考；独立 KL loss，**β=0.001，veRL k3+**；不把 KL 混进奖励和动态采样 |
| 优化 | LR1e−6，15次更新线性预热；LoRA r16/alpha32/dropout0；clip0.2，梯度裁剪1；每批一次更新，动作 token 全局均值 |
| 预算 | **250次有效更新 = 8,000条进入优化的轨迹**，不含过滤和评测；每25步保存并做完整 dev300，共10轮 |
| 选模 | SFT起点与25/50/…/250步共同按严格成功数、Reward v3、较少步数、较早检查点排序；无收益可保留SFT |

配置：[基线配方](../../configs/grpo.json)＋[本版覆盖项](../../configs/grpo_stable.json)。β与批次大小是基于现有证据的初始选择，未宣称全局最优。相对旧终局版本，有效轨迹预算翻倍、更新次数减半；**不是等计算预算消融**，也不能把变化单独归因于KL。

## 3. 损失与实现要点

对所有有效轨迹的真实 assistant 动作 token，以全局 token 总数归一化：

```text
L = mean_action_tokens[clipped_policy_loss(A, πθ/πold) + 0.001 × KL_surrogate(πθ, πS-repair)]
A = training_reward − 同题4条回报均值
```

- **KL有真实梯度。** 使用当前安装 veRL 的 `k3+`：数值报告非负k3估计，反向使用k2的梯度。在本实现单批、单次on-policy更新下，其期望对应条件动作分布的前向KL梯度；不是直接对固定采样的k3求导。已与小词表精确KL的梯度核对。veRL对k3数值含截断，所以记录的是**采样估计，非精确全词表KL**。
- **参考固定。** S-repair权重冻结，新增RL LoRA无bias更新；先在`no_grad + eval + disable_adapter`下计算参考，再恢复actor训练和adapter，才前向/反向。只保留采样token的参考logprob，不常驻第二份参考权重；梯度检查点重算始终使用actor adapter。
- **双卡累积。** 收齐32条后按动作历史长度分配两卡；逐轮计算，梯度按全批动作token数归一化后求和，最后只执行一次优化器更新。采样期间不发布中间策略；原有FLA/SDPA及梯度检查点继续复用。
- **熵只诊断。** 在actor已经生成的logits上，分块计算完整词表分布熵，不增加前向、不反传熵奖励。参考KL、真实熵与采样器/actor数值差异分别命名。它们都是**本次参数更新前、采样状态上的统计**，跨步变化也受状态分布影响。

新增模块仅为[参考约束与熵](../../scripts/grpo/regularization.py)；采样、训练、监督进程继续复用现有入口。方法依据：[veRL GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html)、[veRL KL实现](https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py)；验收清单记录实际安装源码哈希。

## 4. 运行、停止与保存

正式入口如下。在双H100 worker使用已冻结运行环境、FLA安装、已运行的开发机接收/HDFS归档队列；模型和工作目录位于临时盘，地址由环境变量提供：

```bash
# 必须设置 SHOP_RUNTIME、SHOP_GRPO_KERNELS、SHOP_ARTIFACT_QUEUE、SHOP_ARTIFACT_ROOT。
# S_REPAIR 指向已校验的 S-repair step72；新工作目录位于 SHOP_ARTIFACT_ROOT 下。
mkdir -p next/outputs/grpo/stable
nohup setsid bash next/scripts/grpo/launch.sh --variant stable \
  --model "$S_REPAIR" --work "$SHOP_ARTIFACT_ROOT/grpo-stable" \
  --output next/outputs/grpo/stable \
  >next/outputs/grpo/stable/supervisor.log 2>&1 < /dev/null &
```

- **正常停止：250次有效更新完成。** 不因某次dev涨跌临时改变预算。连续10次有界补采仍凑不齐、非有限数值、协议错配、服务退出或开发机备份失败时停止并保留证据。
- 每25步保存adapter、优化器、随机/采样状态和日志；开发机`/tmp`校验接收后继续，独立队列同步HDFS。HDFS配额不足保留开发机副本重试；入口拒绝未配置归档队列的本版训练。
- 训练结束导出选中与最终模型；结果身份以实际评测的**基座＋adapter**为准。源码快照、训练/评测日志、逐题轨迹、TensorBoard随阶段归档。
- 复用已有后台监督与双卡保活恢复，SSH断开不影响执行；可捕获失败亦恢复。worker被回收或强杀无法保证执行清理。保活负载不作为训练吞吐。

结果分目录：`outputs/grpo/stable/`为运行证据；`results/grpo/stable/`为验收/最终清单；本目录为阶段总结。原outcome/local结果不覆盖。

## 5. 如何判断有用

TensorBoard及`metrics.jsonl`记录参考KL、KL loss、policy/total loss、真实动作熵、梯度、有效组、过滤成本和耗时；dev另记录原始奖励、错误购买、循环及相对S-repair的**新增成功/丢失成功**，对应题目ID进入`selection.json`。

目标是严格成功净增，同时核对奖励、错误购买、循环与推理成本是否退化；只变短或训练reward变高不能称为全面提升。沿用既定选模规则，若有指标取舍就如实报告。开发集选模通过后再冻结候选、授权测试；已有test1000已做过多轮对照，后续结果必须披露其复用情况，不称为全新盲测。稳定性和KL单项因果主张仍需后续多种子/匹配预算消融，本轮不预建这些实验。

**交付验收：55项CPU测试通过**，覆盖参考冻结/LoRA恢复及梯度检查点反向、KL正确梯度、完整分布熵、32条分波采样、无效组过滤/不足整批跳过、双卡全局归一化、既有局部信用及执行契约回归。冻结协议及训练/留出隔离已核验。双H100正式启动验收单列在上述清单；首批正常更新只证明计算链路可执行，不证明最终收益或所有长序列都不会触及显存上限，效果按后续dev检验。
