# 本次复现结果（Baseline、SFT、GRPO 已完成）

GRPO 于北京时间 2026-09-29 02:13 完成，2026-09-30 核验结果。仅报告本次复现；原文档和作者历史成绩保留。

## 已核验的最终 200 题结果

| 阶段 | 严格成功 | 严格成功率 | 平均奖励 | 平均步数 |
|---|---:|---:|---:|---:|
| Baseline | 2/200 | 1.00% | -0.12325 | 5.590 |
| SFT | 131/200 | 65.50% | 0.558408 | 9.555 |
| GRPO（第 450 步） | 132/200 | 66.00% | 0.559420 | 10.275 |

SFT 较 Baseline 提高 64.5 个百分点。两组均有 200 条唯一任务轨迹，任务集合与当前
Final-200 完全一致，评测协议一致。本次逐条重新按完整终局条件核算严格成功，
包含 `gold_purchase`、`reward_valid=true`、成功终止与购买条件，计数与汇总一致。

SFT 状态：196 条 `done`、2 条 `invalid_action_limit`、2 条 `error`。
两条 error 均为上下文超过输入预算，仍计入 200 题分母；`done` 不等于严格成功。

## SFT 训练

- 输入 800 条训练轨迹；超长过滤 1 条，实际训练 799 条；验证 200 条。
- 完成 3 个 epoch / 300 个优化步，训练及验证运行约 3 小时 31 分，整体脚本约 214 分钟。
- 总训练 loss：0.311578；显存分配峰值：74.46 GiB。
- 各轮验证 loss：0.343979、0.332419、0.333728。
- 正式 SFT 评测使用按冻结配方完成的第 3 轮模型，未用 Final-200 选点。
- LoRA、100/200/300 步检查点以及合并模型均已保存在共享项目目录。

模型：`outputs/models/sft-lora/`、`outputs/models/sft-merged/`。

## GRPO 完成情况与分析

双卡 H100 完成 500 次参数更新，无非有限损失或梯度；训练分配显存峰值约 59.05 GiB/卡。
第 100 步正常中断后通过 veRL 原生检查点恢复，启用 TensorBoard logger，未从头重新训练。
独立 50 题验证每 50 步执行，检查点实际每 10 步保存，50 个检查点已归档 HDFS。

初始验证平均奖励 0.373448；第 450 步达到 0.487896，第 500 步为 0.472236。
按预先规定的“验证平均奖励最高、同分取较早步数”选择第 450 步；Final-200 不参与选点。
该检查点完成 FSDP 恢复和 LoRA 合并，正式评测覆盖 200 条唯一任务，协议与 SFT 相同。
逐条按完整终局条件重算严格成功，确认 GRPO 132 题、SFT 131 题与汇总一致。

GRPO 较 SFT 净增 1 题（0.5 个百分点），平均奖励仅增加约 0.00101。
逐题对比为 123 题共同成功、9 题新增成功、8 题退化、60 题共同失败；
配对精确 McNemar 检验 p=1.0，未显示严格成功率显著提升。
错误购买从 10 题降至 5 题，但重复循环从 11 题增至 18 题，非法动作超限从 2 题增至 4 题，
平均交互步数增加约 7.5%。验证奖励提升尚未充分转化为稳定的测试收益，不能宣称 GRPO 显著优于 SFT。
GRPO 的 1 条上下文预算错误和 4 条非法动作超限均保留在 200 题分母中。

训练结束和失败后自动恢复双卡保活；2026-09-30 核查两个保活进程仍在运行。
这仅覆盖监督进程可执行清理的情况，Worker 整体回收后无法由进程内脚本恢复。

完整指标、验证候选、评测协议与原始证据哈希见 [GRPO 结果数据](grpo-results.json)；
配置、命令及运行限制见 [GRPO 复现说明](grpo.md)。外部模型裁判未调用，本次分数来自 Reward v3。

## 证据

- `outputs/evaluation/baseline/{summary.json,trajectories.jsonl,report.html}`
- `outputs/evaluation/sft/{summary.json,trajectories.jsonl,report.html}`
- `outputs/models/sft-lora/train_summary.json`
- `outputs/models/sft-merged/merge_manifest.json`
- `outputs/reproduction/grpo-dual-h100/{train-tensorboard.log,training_diagnostics.jsonl}`
- `outputs/reproduction/grpo-dual-h100/evaluation/{summary.json,trajectories.jsonl,report.html}`
- `outputs/reproduction/grpo-dual-h100/{checkpoint-selection.json,run_manifest.json}`
- `outputs/reproduction/results_check_2026-09-21.json`
- `outputs/reproduction/run_manifest.json`

作者归档中的 SFT 60.5%、GRPO 62.0% 不作为本次成绩，数据与协议差异也不支持据此作直接优劣结论。
