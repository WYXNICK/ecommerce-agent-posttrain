# 冷启动 SFT：Astra → Qwen3.5-2B（S0）

## 目的与数据

从与新版 Baseline 相同的固定原始模型训练领域 S0，学会真实购物环境的工具协议与购买策略。正式教师为 **gpt-6-astra-2026-09-03**；Sol 数据保持独立，不混用。

- 输入：[主数据选择](../../configs/data_selection.json)、[Astra 成品清单](../../data/teacher/astra/manifest.json)。**2,000 条训练＋200 条独立验证**，全部为严格成功终局。
- 每条完整轨迹使用冻结的学生模板、`enable_thinking=false`，只监督 assistant（含工具调用）；系统、用户、工具返回全部 mask。不监督教师私有推理。
- 全量重算序列长度和监督 token 数，必须逐条与采集成品一致；不截断、不静默丢弃。训练与验证、开发、测试及历史留出任务隔离。

## 实现与固定配方

入口：[supervise_sft.py](../../scripts/sft/supervise_sft.py)；权威参数：[sft.json](../../configs/sft.json)。原环境、执行协议、奖励和历史结果保持不变；新增训练代码、数据与权重的 SHA-256 写入本次 `run.json`。原始权重哈希必须与已完成 Baseline 的模型一致，错误起点会拒绝启动。

| 项目 | 设置 |
|---|---|
| 原始学生 | Qwen/Qwen3.5-2B；revision `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| 训练 | 双 H100，BF16 LoRA＋DDP；3 epoch＝750 次更新，seed 42 |
| 批次 | 每卡 1 条完整轨迹，累积 4 次；有效批次 8 |
| 优化 | LR 1e-4，线性衰减，warmup 3%，AdamW，梯度裁剪 1.0 |
| LoRA | r=16、alpha=32、dropout=0.05；注意力及 MLP 投影，具体模块见参数文件 |
| 长序列 | 上限沿用 24,576；BF16、梯度检查点、SDPA；按长度分组采样 |
| 输出投影 | 只计算有监督位置的词表 logits；与完整 masked CE 的 loss、梯度及跨卡累积 token 归一化等价，已有测试 |
| 保存与验证 | 每 125 步保存可恢复检查点；每 epoch 计算 200 条验证 loss |

## 选模与验收

训练后自动 CPU 合并第 **250/500/750 步** LoRA，逐个在**完整 300 题开发集**执行同一冻结协议。按严格成功数→平均 Reward v3→较少动作→较早检查点选 S0。验证 loss 只作诊断；**本阶段不运行 1,000 题测试，不承诺固定成功率**。GRPO 与失败回流另行执行。

训练前检查：完整数据与留出校验；masked loss 和梯度等价；不同 token 数的跨卡梯度累积；真实 Qwen3.5 稀疏投影和 LoRA 反传。正式训练确认有限 loss/梯度、双卡实际显存及利用率后交后台运行。

## 运行、监控与归档

- **状态：2026-11-02 已启动正式训练，计划 750 次更新；最终成绩待训练及开发集选模完成。** 当前未产生可报告的 S0 成功率。
- 本地运行目录：`next/outputs/sft/astra-s0/`；只看 `status.json`（阶段）、`progress.json`（训练步数/ETA）、`train.log`（训练日志）。`run.log` 是后台监督入口。
- 原生 TensorBoard：`tensorboard --logdir next/outputs/sft/astra-s0/tensorboard --host 127.0.0.1 --port 6016`，当前已在开发机 `127.0.0.1:6016` 启动，SSH 转发该端口可查看 loss、验证 loss、学习率及梯度；事件目录保留。已有服务运行时无需重复启动。
- `gpu.jsonl` 与 `gpu-0.json` / `gpu-1.json` 分别记录 `nvidia-smi` GPU、dmon SM 利用率及一分钟均值，目标 **GPU >30%、SM >20%**。初始化、保存与切换阶段可短时下降；持续不足会记录告警。训练效率根据实际监控验证。
- 缓存、模型及检查点放 worker `/tmp`；小体积日志、开发集逐题轨迹与清单放共享项目目录。训练入口独立会话运行，SSH 断开不影响；worker 本身被销毁仍会中断。
- 完成或失败后监督进程自动恢复 GPU 0/1 保活；CPU 准备和导出阶段保持保活，开发集推理使用 GPU 0、GPU 1 保活。
- 选模后归档选中独立模型、选中和最后可恢复检查点、全部开发轨迹及训练指标到启动时指定的 HDFS 路径；`archive.json` 记录状态与哈希。配额等归档失败明确标记 pending，保留本地成果，不能当作归档成功。
- 阶段完成的权威结果写入 `next/results/sft/astra-s0/summary.json`；历史 SFT/E1/E2/E3 不混入新版结果。

复现需先激活冻结运行环境，再执行入口的 `--help` 并提供模型、worker 临时目录、共享输出、既有保活脚本和 HDFS 归档地址；这些机器路径保留在本次运行清单中，不固化在源码。
