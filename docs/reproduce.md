# 运行、恢复与继续开发

当前入口全部在 `next/`；根目录的 `scripts/train_*.sh`、旧 `shopping-grpo` CLI 不对应本文主结果。下面是执行契约与命令模板，变量需指向已准备并校验的实际目录。不要为查看结果重复训练。

## 1. 环境与数据

运行使用 Linux、Python 3.11、两张 H100 80GB。关键版本：PyTorch 2.11.0、veRL 0.8.0、vLLM 0.25.1、PEFT 0.20.0、Transformers `7ea2320c76117e6742364808a666ef6f2fb40a67`。共享执行依赖以 [execution_manifest.json](../next/configs/execution_manifest.json) 为准，训练依赖还需核对对应 `run.json`；不要用“最新版”替代冻结版本。

需要准备：

1. 学生 revision 权重及 tokenizer，商品数据与搜索索引；来源及数据哈希见 [metadata.json](../next/data/task_pool/metadata.json)。商品压缩包位于环境目录，解压与索引构建逻辑见 [setup.sh](../scripts/setup.sh)；该脚本也安装早期执行栈，不能单独代表当前运行验收。
2. Python 运行环境与 ShopSimulator 独立环境。当前入口可使用 `SHOP_PYTHON` 指定环境解释器；模型、缓存、临时工作区优先放 `/tmp`，长期成品放外部存储。
3. 从[模型清单](../next/results/evaluation/final-candidates/manifest.json)恢复目标基座及adapter，校验文件哈希。只恢复需要的模型，不下载全部历史权重。
4. 教师接口需要 ModelHub 访问与凭证；凭证仅从环境变量/私有文件读取。已有验收数据可用于SFT而无需再次调用教师。网关不对所有外部用户开放，其他提供商需适配客户端并建立新的执行清单。

```bash
# 已激活冻结Python环境；从仓库根目录执行。
export PYTHONPATH="$PWD/next/src:$PWD/next/scripts/sft:$PWD/next/scripts/grpo"
python next/scripts/agent.py check
python -m unittest discover -s next/tests -v
```

完整测试还会导入ShopSimulator的Flask等依赖，需在测试环境安装环境目录的requirements；训练与环境服务在正式运行中可使用独立解释器。离线校验应设置 `HF_HOME` 指向已缓存的固定revision，必要时使用 `HF_HUB_OFFLINE=1`，否则tokenizer检查会尝试联网。

`check`核对数据、源码、依赖及模板；报错应先核查身份。只有有意修改共享协议、审阅且准备新运行时才使用`agent.py freeze`，不能用重新冻结掩盖旧实验漂移。

完整采集、环境启动与学生单模型评测命令见[执行协议](../next/docs/execution.md)。

## 2. SFT与纠错

参数文件：[sft.json](../next/configs/sft.json)、[reflow.json](../next/configs/reflow.json)。成品选择由 [data_selection.json](../next/configs/data_selection.json) 固定。

```bash
python next/scripts/sft/run_pipeline.py   --model "$BASE_MODEL" --work "$WORK" --output "$OUTPUT"   --keep-on "$KEEP_ON" --keep-off "$KEEP_OFF" --archive "$HDFS_ROOT"
```

需先配置下节的工件接收服务及 `SHOP_ARTIFACT_QUEUE`。入口串联冷启动、dev选模、普通/修复增量对照；只监督动作/纠错后缀。参数及保存说明见[SFT](../next/docs/sft/README.md)、[纠错](../next/docs/data-reflow/README.md)。

## 3. GRPO

主线配方为 [grpo.json](../next/configs/grpo.json)。`SHOP_RUNTIME` 指向含 `activate.sh` 的冻结环境目录；`SHOP_GRPO_KERNELS` 指向 `site/` 和 `wheels/` 均齐全的训练算子目录，版本见 [requirements.txt](../next/scripts/grpo/requirements.txt)。先完成工件接收设置，再运行：

```bash
bash next/scripts/grpo/launch.sh   --variant outcome --model "$REPAIR_SFT"   --work "$WORK" --output "$OUTPUT"
```

训练500次有效更新，每50步保存和完整dev300评测；起点参与选模。结束或可捕获失败后恢复GPU保活。后台执行可在外层使用`nohup`或独立会话，标准输出重定向到独立监督日志；训练主日志由入口管理。断SSH不等于worker重启保护。

`--resume "$CHECKPOINT"` 恢复检查点；完整adapter、优化器、随机状态和采样位置均需保留，使用独立输出目录并遵守源码/配方校验。不要将不同基座或修改后的配方当成原实验的连续更新。

## 4. 备份与监控

在worker配置 `SHOP_ARTIFACT_ROOT`（所有模型与运行工件的父目录）和 `SHOP_ARTIFACT_QUEUE`（worker及开发机可访问的队列）。开发机接收服务示意：

```bash
python next/scripts/sft/artifacts.py receive   --queue "$QUEUE" --destination "$BACKUP"   --worker "$WORKER_SSH" --port "$SSH_PORT" --worker-root "$WORK"
python next/scripts/sft/artifacts.py archive   --queue "$QUEUE" --destination "$BACKUP" --hdfs "$HDFS_ROOT"
```

两者是常驻服务，分别在独立会话启动。`--worker-root` 必须等于worker的 `SHOP_ARTIFACT_ROOT`，不能误用某个子目录。检查点写完后接收端核对SHA256和大小并发回执，外部上传独立执行；没有接收器会阻塞保存。恢复时再次按清单校验，而不是只看文件名。

看 `status.json` / `progress.json` 判断阶段状态，`train.log`排错，`metrics.jsonl`绘制逐更新指标，`dev/step-*/`查看选模。TensorBoard：

```bash
tensorboard --logdir "$OUTPUT/tensorboard" --host 127.0.0.1 --port 6006
```

GPU保活只用于资源管理，不计入有效训练利用率；训练充分占用时应让出计算。机器被销毁或SIGKILL不能靠退出清理保证恢复，外部备份仍是恢复依据。

## 5. 固定候选评测

使用 [test_comparison.json](../next/configs/test_comparison.json) 中的精确基座＋adapter及权重哈希恢复目录：

```bash
python next/scripts/evaluation/run.py   --models-root "$MODELS_ROOT" --work "$EVAL_WORK" --output "$EVAL_OUTPUT"   --plan next/configs/test_comparison.json
```

该方案还引用已完成Baseline/教师的逐题证据，恢复这些来源后才能复用；不能只把一个权重目录传入就声称完成原对照。新候选使用独立评测方案和目录。多次采样诊断另见 [pass@k方法](../next/docs/passk/README.md)，不能用单次成绩外推pass@8。

## 6. 后续修改Reward

- 在 `next/scripts/grpo/` 和 `next/configs/` 定义新实验，复用 `settings.py → rollout.py → train.py`，不修改已经发布结果的配方或原始证据。
- 把“是否可训练”“training_reward”“动态采样资格”“优势”一起检查；评测仍保持Reward v3和严格成功口径。基于明确结构化证据判定，不把未知直接当违约。
- 针对真实边界运行对应 `test_*.py`；共享执行协议未改时不重新冻结它。训练算法变更由新 `run.json` 的源码/配置哈希记录。
- 新配方独立初始化，先dev选模；保留起点、选中检查点和完整日志。不要把新目标训练混入旧日志称为原实验续训。

当前约束奖励与成功优先实现都保留，分别见[约束设计](../next/docs/grpo-constraints/README.md)、[成功优先设计](../next/docs/grpo-success/README.md)。它们是可继续研究的分支，并非已证实超过主GRPO的方案。
