# 本次 GRPO 复现

本说明对应本次双卡 H100 完成的 GRPO，原作者文档保留。最终指标与证据索引见
[结果](results.md) 和 [结果数据](grpo-results.json)。

## 配方与实现

- ShopSimulator v2.1、Reward v3、Observation v2、工具 Schema v2。
- 以本次 SFT 合并模型为起点，新建 LoRA：rank 16、alpha 32、all-linear；视觉塔冻结。
- 1000 条 GRPO 训练任务、50 条独立验证任务，Final-200 不参与训练或选点。
- 两张 H100 80GB、单节点 FSDP；参数与优化器 offload，BF16、SDPA、梯度检查点。
- 全局每批 2 个问题，每题 4 条采样轨迹；每卡 micro batch 为 1。
- prompt 上限 4096、response 上限 20480，完整序列上限 24576，未以缩短轨迹解决显存问题。
- 学习率 1e-6、warmup 比例 0.03、500 次参数更新；每 50 步验证。
- 动态采样过滤组内奖励相同的组，补采预算从 3 批增至 8 批；这是本次配方调整。
- 熵系数为 0，关闭不参与目标的熵统计。对已有 log-softmax + gather 使用
  `torch.compile(dynamic=True)` 融合前向与反向，减少完整词表中间张量；策略目标保持不变。
- `train_grpo.py` 支持原生 TensorBoard 和显式检查点恢复，预检使用实际训练的配置及覆盖参数。
- `export_grpo.py` 先恢复 FSDP 权重，再合并独立 LoRA，避免只发布冻结的 SFT base；
  对缺失适配器权重或无效 alpha 拒绝发布。

本次运行覆盖配置中的保存间隔，将 `trainer.save_freq` 从 50 改为 10；验证间隔仍为 50。
第 100 步恢复时关闭重复的训练前验证，原有第 50、100 步验证记录保留。
恢复包含模型、优化器、调度器、随机状态和 `data.pt` 数据迭代状态。

## 运行命令

以下命令在已经安装锁定 GRPO 依赖、能访问 SFT 合并权重和 ShopSimulator 服务的环境执行。
路径通过变量指定；环境服务必须与训练进程的地址一致。

```bash
export CUDA_VISIBLE_DEVICES=0,1
# 本次 Ray 环境需要显式按所分配的 local rank 选卡。
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export NCCL_CUMEM_ENABLE=0
export SFT_MODEL=/path/to/sft-merged
export GRPO_OUTPUT=/path/to/grpo-checkpoints

python scripts/train_grpo.py \
  --model "$SFT_MODEL" --output "$GRPO_OUTPUT" \
  --logger tensorboard --experiment-name shopping-grpo-dual-h100 \
  -- +ray_kwargs.ray_init._node_ip_address=127.0.0.1 trainer.save_freq=10
```

`GRPO_OUTPUT` 对新训练须为新目录或空目录。恢复时明确指定完整检查点：

```bash
python scripts/train_grpo.py \
  --model "$SFT_MODEL" --output "$GRPO_OUTPUT" \
  --logger tensorboard --experiment-name shopping-grpo-dual-h100 \
  --resume-from "$GRPO_OUTPUT/global_step_100" \
  -- +ray_kwargs.ray_init._node_ip_address=127.0.0.1 \
  trainer.save_freq=10 trainer.val_before_train=false
```

恢复前保留原诊断记录；若中断时产生了检查点步数以后的记录，将这些记录另存，
避免重算步数混入正式诊断。恢复不应删除已经完成的验证记录。

原生 logger 写入 `$GRPO_OUTPUT/tensorboard`；通过本地端口转发访问：

```bash
tensorboard --logdir "$GRPO_OUTPUT/tensorboard" \
  --host 127.0.0.1 --port 6006 --reload_interval 15
```

本次看板的 `live` 为恢复后的原生记录；`history_steps_1_100` 为原始标量的一次性补录。
历史补录应按 Step 横轴查看，写入时间不代表原始训练时间。
关注独立验证奖励、更新步数、梯度、有效采样组比例和显存；训练采样奖励不等于验证成绩。
`actor/ppo_kl` 不是相对最初 SFT 模型的完整漂移指标；本次不计算熵。

仅按独立验证平均奖励选择模型，同分取较早步数；本次选中第 450 步：

```bash
python scripts/export_grpo.py \
  "$GRPO_OUTPUT/global_step_450/actor" /path/to/grpo-merged
bash scripts/serve_model.sh /path/to/grpo-merged
# 在模型服务就绪后，于另一终端执行正式评测。
EVAL_OUTPUT_DIR=outputs/evaluation/grpo bash scripts/evaluate.sh grpo
```

## 后台执行、归档与保活

本次监督进程独立于 SSH 会话，检查点写入 Worker 临时盘，仅在训练器完成保存标记后上传 HDFS。
清理只针对本次启动进程的后代，结束或失败后恢复双卡原保活脚本。
执行期间使用独立低占空比负载降低资源回收风险；其 SM 活跃度不能计作训练效率。
Worker 整体被回收或进程遭 SIGKILL 时，无法保证进程内清理执行。

机器相关启动、监督、保活和看板布局文件保存在运行证据中，不加入项目通用启动入口。
依赖 CUDA 13 的 Torch 运行在旧驱动上需要对应 CUDA compatibility 库；本次仅在运行环境设置，
未更换系统驱动，也未将机器路径写入项目代码。

模型和 50 个检查点的 HDFS 根目录：

```text
hdfs://harunava/home/byte_data_tt_poi/user/wangyuxuan/shopping-grpo-longhorizon/grpo-dual-h100-20260928
```

`global_step_450/` 是选中的恢复检查点，`grpo-merged/` 是正式评测模型，`logs/` 保存运行证据。
本地证据入口为 `outputs/reproduction/grpo-dual-h100/`：

| 文件 | 用途 |
|---|---|
| `status.json` | 最终阶段和双卡保活恢复结果 |
| `train-tensorboard.log` | 恢复后原始训练输出 |
| `training_diagnostics.jsonl` | 完整训练标量和采样轨迹 |
| `checkpoint-selection.json` | 10 次验证及选点依据 |
| `evaluation/report.html` | 正式 200 题评测报告 |
| `run_manifest.json` | 数据、源码哈希及运行配置 |

## 验证

完整 Worker 单元测试 207 项：206 通过、1 跳过。原生 logger 与恢复入口的针对性测试通过。
双卡实际 24576 token 前向、反向及优化器测试的峰值约 64.74 GiB/卡；
正式 500 次更新的分配显存峰值约 59.05 GiB/卡，未出现非有限损失或梯度。
编译融合对照中 BF16 输出差为 0，梯度相对误差约 6.08e-5；FP32 最大输出差约 9.54e-7。
实际第 450 步模型已完成 FSDP 恢复、LoRA 合并、HDFS 归档和完整 200 题评测。
