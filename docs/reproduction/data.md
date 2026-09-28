# 本次复现：固定 SFT 数据

## 数据与来源

| 用途 | 文件 | 条数 |
|---|---|---:|
| SFT 训练轨迹 | `data/sft/train.jsonl` | 800 |
| SFT 验证轨迹 | `data/sft/validation.jsonl` | 200 |
| 最终评测任务 | `data/evaluation/tasks.jsonl` | 200 |

SFT 验证集与最终评测集是两套独立数据，三者 task ID 互不重叠。本次使用固定划分，没有重新采集或重划分。文件 SHA-256 见 [结果数据](sft-results.json)。

仓库 [metadata.json](../../data/sft/metadata.json) 标注教师为 `deepseek-v4-flash`：原始采集 2,498 条，通过筛选 1,026 条，其中固定训练 / 验证使用 1,000 条，余下 26 条未使用。作者提供的评论说明新数据使用正式版、原报告训练结果尚未更新；metadata 未给出教师模型快照版本，因此不能独立确认具体服务端权重版本。

## 质量与预处理

已有教师动作回放记录显示，在冻结环境中 1,000 / 1,000 条满足严格成功条件。这是教师动作回放结果，不是 SFT 模型推理成功率。

训练仅监督 assistant token，user 和 tool token 不计算训练损失。长度上限为 24,576 token，超长轨迹整条丢弃，不截断终局动作。实际保留训练 799 条、验证 200 条；训练丢弃 1 条。保留训练轨迹共 4,809,823 token，其中监督 token 为 268,942。

环境契约固定为 Environment v2.1、Reward v3、observation v2、tool schema v2。严格成功要求完整 `gold_purchase` 终局且 `reward_valid=true`。数据回放统计、预处理计数和训练实际输入数量均收录于 [结果数据](sft-results.json)。
