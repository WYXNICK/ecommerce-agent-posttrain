# 训练、评测与模型保存

## 执行顺序

1. 固定原始 Qwen3.5-2B，使用 Astra 2,000 条训练、200 条验证，执行双卡 LoRA SFT：3 epoch、750 次更新。
2. 合并每个 epoch 的模型，分别执行完整 dev300，按成功数、奖励、步骤、较早检查点选择 S0。
3. 从同一个 S0 分别训练普通增量组与修复组：各 144 条新增示范＋相同 144 条原数据，2 epoch、72 次更新。直接读取冻结成品，修复组仅监督教师后缀。
4. 两组各评测两个 epoch，与 S0 一起选下一阶段起点；随后才进入 GRPO。学生最终 test1000 本阶段不运行。

入口为 `scripts/sft/run_pipeline.py`；参数分别以 `configs/sft.json`、`configs/reflow.json` 为准。基础执行协议、数据划分、奖励及评测口径不变。

## 只看这几个入口

- `next/outputs/pipeline/current.json`：当前运行目录、worker、开发机模型目录及 HDFS 地址。
- 当前运行目录的 `status.json`：整个流水线；`sft/status.json`、`reflow/status.json`：具体阶段。
- `sft/progress.json`、`reflow/{general,repair}/sft/progress.json`：训练步数、loss、ETA；对应 `tensorboard/` 为原生看板事件。
- `storage/receipts/`：开发机完整保存和逐文件 SHA-256 验证回执；`storage/hdfs/`：HDFS 完成回执。`archive-status.json` 记录配额阻塞。
- 阶段结果统一发布到 `results/sft/astra-s0/summary.json`、`results/data-reflow/r1/summary.json`；不同运行的模型与指标不混用。历史证据集中在 `results/history/`、本地 `outputs/history/`。

查看看板：`tensorboard --logdir next/outputs/pipeline/20261104 --host 127.0.0.1 --port 6016`，通过 SSH 转发该端口。

## 保存与资源规则

- **开发机是主保存端。** 原始模型、每个已保存检查点（含优化器、调度器和双 rank RNG）、最终 adapter、导出模型都保存在开发机 `/tmp`。worker 保留运行副本。
- **保存完成才继续。** 所有 rank 完成检查点写入后，开发机接收、逐文件核对字节数及 SHA-256、原子提交目录，再确认保存。导出模型同样先保存再评测。不会把正在写入的半成品当作备份。
- **HDFS 自动补备份。** 开发机后台服务检测已保存工件；正常每分钟发现新工件，配额等失败后每 5 分钟重试。成功上传并核对文件清单/大小后发布完成目录。训练不等待团队配额恢复；本地 `/tmp` 也不是永久存储。
- **有效计算优先。** 空闲阶段使用既有双卡保活。训练/推理阶段先停满负载保活，采用小显存、短 kernel 的补载进程：自身暂停后采样真实 GPU 活动，低于阈值时短时补载，正常计算充足时不提交补载。不同 CUDA 进程之间不保证严格抢占，不能承诺零竞争或每个瞬间利用率达标。
- 记录 GPU 与 dmon SM 的一分钟均值，目标分别 >30% / >20%；低利用率继续报警。保活产生的负载不计为训练吞吐，训练效率看真实步速与有效计算。正常退出或可捕获异常后恢复原双卡持续保活。
- 所有监督和保存服务独立会话运行，SSH 断开不影响；worker 被平台销毁无法靠脚本保活。已经同步的工件仍在开发机，下一次可按保存状态恢复。
