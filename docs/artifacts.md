# 结果、模型与日志索引

## 保存层次

| 内容 | 权威入口 | 保存方式 |
|---|---|---|
| 主结果、逐题成功ID、配对差异 | [test1000](../next/results/evaluation/test-comparison/summary.json)、[pass@k](../next/results/passk/dev300/summary.json) | Git |
| 数据划分、来源、隔离与哈希 | [任务清单](../next/data/task_pool/metadata.json) | Git；大商品数据依清单恢复 |
| 教师与纠错成品 | [Astra](../next/data/teacher/astra/manifest.json)、[Sol](../next/data/teacher/sol/manifest.json)、[纠错](../next/data/reflow/r1/manifest.json) | 成品及清单保留 |
| 候选模型身份与文件哈希 | [候选清单](../next/results/evaluation/final-candidates/manifest.json) | Git中的索引＋外部权重 |
| 每轮配置、源码、日志、逐题轨迹 | 对应运行的 `run.json`、`evidence/`、`rollouts/`、`dev/`、`train.log`、`metrics.jsonl`、`tensorboard/` | `next/outputs/`及外部归档 |
| 已确认保存的工件 | 各阶段 `archive.json`、归档目录 `artifact-manifest.json` | Git摘要＋外部逐文件清单 |
| 历史成果 | [历史实验](reproduction/README.md)、[历史运行](../next/results/history/20261103/index.json) | 独立保留，不能混入当前模型身份 |

`outputs/` 是本地运行区，不随clone出现；不要把Git里的summary误认为全部模型或原始日志。仓库重命名不修改历史工件URI、源码快照或哈希。

## 按阶段查找

| 阶段 | 指标/证据索引 | 归档记录 |
|---|---|---|
| Baseline | [结果](../next/results/baseline/summary.json) | 结果中的原始来源与哈希 |
| 教师采集/评测 | [Astra采集](../next/results/teacher-collection/astra/summary.json)、[Sol采集](../next/results/teacher-collection/sol/summary.json)、[评测比较](../next/results/teacher-evaluation/comparison.json) | [教师开发评测](../next/results/teacher-evaluation/astra/dev/archive.json)，其余见阶段来源 |
| S0 | [结果](../next/results/sft/astra-s0/summary.json) | [候选保存](../next/docs/evaluation/final-candidates.md) |
| 纠错与普通增量SFT | [结果](../next/results/data-reflow/r1/summary.json) | [回执](../next/results/data-reflow/r1/archive.json) |
| 主终局GRPO | [结果](../next/results/grpo/outcome/summary.json)、[分析](../next/results/grpo/outcome/analysis.json) | [回执](../next/results/grpo/outcome/archive.json) |
| S0直接GRPO | [结果](../next/results/grpo/s0/summary.json) | [训练](../next/results/grpo/s0/archive.json)、[测试](../next/results/evaluation/s0-grpo-test/archive.json) |
| Batch / 局部信用 / 稳定终局 | [batch](../next/results/grpo/batch/summary.json)、[local](../next/results/grpo/local/summary.json)、[stable](../next/results/grpo/stable/summary.json) | [batch](../next/results/grpo/batch/archive.json)、[local](../next/results/grpo/local/archive.json)；stable见结果中的保存证据 |
| SDPO | [R1](../next/results/sdpo/r1/summary.json)、[R2](../next/results/sdpo/r2/summary.json) | [R1](../next/results/sdpo/r1/archive.json)、[R2](../next/results/sdpo/r2/archive.json) |
| 证据约束GRPO | [结果](../next/results/grpo/constraints/summary.json) | [500测试](../next/results/evaluation/grpo-constraints-test/archive.json)、[450测试](../next/results/evaluation/grpo-constraints-450-test/archive.json)；训练保存见阶段结果 |
| 成功优先GRPO | [结果](../next/results/grpo/success/summary.json) | [测试回执](../next/results/evaluation/grpo-success-test/archive.json)；训练保存见阶段结果 |
| 四模型采样诊断 | [结果](../next/results/passk/dev300/summary.json) | [回执](../next/results/passk/dev300/archive.json) |

完整机读文件目录见 [next/results](../next/results/)。归档状态以具体回执为准；索引存在不等于此刻外部存储可访问，也不代表未列入清单的文件已保存。

## 恢复与复盘

1. 先从实验的 `run.json` / 模型清单确定基座、adapter、协议和配置，不能只按step编号认模型。
2. 按归档回执恢复需要的目录，核对 `artifact-manifest.json` 的文件大小和SHA256；精确评测优先基座＋adapter。
3. 训练续跑还需优化器、随机状态和采样位置；只剩合并权重时可推理或新训练，不能声称精确续训。
4. 分析训练曲线使用 `metrics.jsonl` / TensorBoard；逐题失误和奖励分析使用原始rollout与dev/test轨迹，不从汇总分数猜原因。
5. 路径/清理/恢复细节见[存储说明](../next/docs/storage/README.md)。没有验证备份前不清理证据；本次文档整理没有移动模型、训练日志或改写结果JSON。

## 公开使用范围

当前仓库为私有，HDFS与教师网关需要授权。公开发布前应提供可访问的模型/轨迹工件及校验清单，确认数据和依赖的分发许可；本项目不把私有存储链接描述为公开下载地址。凭证不入Git，原始日志分享前检查是否含请求信息或个人路径。
