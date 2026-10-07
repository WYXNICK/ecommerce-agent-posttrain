# GRPO E2：生成预算与真实 token 执行一致性

E2 是从同一 SFT 合并权重开始的无 KL GRPO 控制组。保留 E1 的失败分类、
`-0.65` 非终局模型失败成本、训练奖励筛组和原始 Reward v3 验证选点。
环境、任务划分、学习率、组大小和 500 次更新配方不变。
原有 [E1 实现](grpo-e1.md) 与 [E1 结果](grpo-e1-results.md) 保留。
本页描述已执行的实现与验收。[结果与分析](grpo-e2-results.md) 单独归档为补充备份；
主结果仍沿用原 Chat 评测协议，E2 没有证明严格成功率进一步提升。

## 预算契约

| 边界 | 训练、原生验证与 E2 本地评测 |
|---|---|
| 初始 prompt | 不超过 4096 token |
| 单轮模型输出 | 最多 512；上游缺少 `max_tokens` 时也主动设置 |
| 累计 response 区间 | 20480，包含模型输出和工具消息模板 token |
| 完整上下文 | 24576，保留 512 安全余量及下一轮 512 输出空间 |
| 历史压缩 | 关闭 |
| 工具步数 / 连续守卫拒绝 | 35 / 3 |

两条执行入口调用同一个 `generation_token_limit`。最后一轮输出还受剩余
response 空间限制。模型输出达到 response 上限时不再执行该轮工具；
非终局工具反馈达到上限时不启动下一次生成。已经完成的环境终局优先结算。
预算失败仍属于可训练的模型非终局，不制造有效购买或改写环境分数。

## 工具定义与 token 边界

`environment/tools.py` 是工具定义的权威来源；veRL 工具配置只选择工具名称，
`ShopSimulatorTool` 从该来源构造实际 schema，避免训练与评测使用不同描述。
两侧 raw-token 入口均经过 veRL 的类型化 schema 序列化；该类型不保留
`additionalProperties: false`，因此模型看到的 schema 与旧 Chat 入口存在差异。
运行时参数守卫仍检查额外参数，工具描述与环境契约未改变。

训练保留实际采样的 assistant token。工具反馈使用完整 Qwen 模板边界，
不再受到初始 prompt 的 4096 token 左截断；observation 内容仍先按既有预算投影。

E2 本地评测通过 vLLM `/v1/completions` 请求和返回 token IDs，保留原始生成，
复用 veRL 的 `qwen3_coder` 解析器与工具反馈模板。多工具调用、参数守卫和终止规则
继续使用项目共享实现。工具 token 只作为上下文，学习 mask 为 0；模型生成为 1，
包括模型自身的错误。评测采样仍为 temperature 0、top_p 1。

```bash
EVAL_TOKENIZER_PATH="$LOCAL_MERGED_MODEL" \
EVAL_OUTPUT_DIR="$NEW_EVALUATION_DIR" bash scripts/evaluate.sh model
```

直接调用 benchmark CLI 时用 `--tokenizer "$LOCAL_MERGED_MODEL" --response-length 20480`。
不带 tokenizer 的远端 Chat API 入口不保证原始 token 前缀一致，不能混入 E2 对照。

## 诊断与验收

逐轨迹保存 `generation_turns`：请求前缀 token 数与 SHA-256、有效单轮上限、
剩余响应/上下文空间、实际模型输出、工具 token、累计响应长度。
`tool_tokens_attempted` 记录反馈原长度，`tool_tokens` 只计实际进入轨迹的内容。
原生 TensorBoard 增加模型、工具、累计响应 token 和预算失败率指标。

`tests/test_grpo_token_contract.py` 使用本地真实 Qwen tokenizer/processor，
验证多轮工具调用、非规范空白与 EOS、非法动作恢复时的完整前缀和请求上限一致，
同时验证 mask 对齐、长工具反馈模板完整性和两个 response 边界。

```bash
SHOPPING_TOKENIZER_PATH="$LOCAL_SFT_MODEL" PYTHONPATH=src:tests \
python -m unittest test_grpo_token_contract -v
```

目标双 H100 Worker 已通过 98 项相关回归（包含真实 tokenizer/processor 测试），
以及 5 条独立验证任务的真实模型 API 验收。前 10 次实际更新、token 契约、
完整检查点和 HDFS 副本的启动验收证据保存于本轮输出目录的
`startup-verification.json`。完整训练的 500 次更新与四模型各 200 题重评均已完成，
摘要、逐题成功 ID 和证据哈希见 [结果数据](grpo-e2-results.json)。

## 本轮后台执行

训练从空输出目录与同一 SFT 权重开始；不恢复 E1 的检查点。
500 次实际更新，每 10 步保存、每 50 步验证，按独立验证平均原始 Reward v3
选点，同分取较早步数。Final-200 不参与训练或选点。

后台使用本轮冻结的源码快照、原生 TensorBoard、独立会话监督和 HDFS 归档。
模型、检查点与依赖放在 Worker 数据盘，日志入口保存于
`outputs/reproduction/grpo-e2-20261004/`。`status.json` 为状态入口，
`train.log` 为原始输出，`training_diagnostics.jsonl` 为采样与更新记录。
正常结束、子任务失败或收到 SIGTERM/SIGINT 后清理本轮子进程并恢复双卡保活；
训练期低占空比保活不计作训练吞吐。

选点后自动导出模型，并在冻结的 E2 评测入口下分别评估 E2、SFT、
原 GRPO step 450 和 E1 step 500。结果独立保存于本轮 `evaluation/`，
不覆盖历史结果。任务集合相同，但执行口径修正后的分数不能与旧口径直接混用。

实际训练完成后，首个监督流程因 HDFS 配额异常在归档阶段退出；后处理从已保存的
检查点继续，未重新训练，完成导出、四模型评测与双卡保活恢复。
最终状态为 `completed_local_archive_pending`：本地结果完整，HDFS 选中检查点、
合并模型和最终证据仍待上传；已成功归档的检查点为 step 10–260。
