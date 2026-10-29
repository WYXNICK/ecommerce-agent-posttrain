# 教师正式评测

## 目的与协议

在与学生相同的冻结 1,000 道测试题上测量教师 Agent 的实际能力，作为能力、失败模式和 API 成本的参考。该成绩与训练候选采集成功率分开，不能用于换测试题、修改协议或回流测试数据。

| 项目 | 固定设置 |
|---|---|
| 教师 | Sol：`gpt-6.1-sol-2026-09-29`；Astra：`gpt-6-astra-2026-09-03` |
| 推理与动作 | 沿用采集的 high 推理、网关默认采样和严格 JSON 动作适配，不更换提示或模型快照 |
| 环境 | ShopSimulator v2.1 / Reward v3 / observation v2 / tools v2，共享原执行循环 |
| 可见预算 | 35 步；学生 tokenizer 上下文 24,576、累计 response 20,480、每轮 assistant 512 |
| 单题尝试 | 一次有效模型执行；所有模型失败计入固定 1,000 题分母，不按成功筛选，不行为重采 |
| 基础设施异常 | 请求有界重试；仍失败则保存证据、停止新增题。续跑只补缺失/基础设施异常，不重做已有模型失败 |
| 严格成功 | 完整终局 `gold_purchase`、`reward_valid=true`、`purchase_success=true` |
| 并发 | 每教师 6 路、独立 8 槽环境；不占 GPU |

教师使用 API 私有推理，学生 Baseline 为关闭 thinking 的确定性本地推理；可见执行协议一致不等于计算成本一致。教师结果属于能力参考，不是训练方法的公平消融，也不是学生理论上限。usage 记录已报告 token，未返回 usage 的重试费用不当作已完整计费。

## 本次执行安排

- **Sol：立即执行正式 1,000 题。** 其 2,000＋200 条采集成品已完成。
- **Astra：后台等待完整采集完成后自动执行。** 必须同时满足监督状态 complete、训练 2,000、验证 200 且身份/协议一致；采集失败或部分完成不会触发评测，等待进程保持等待。
- 两个教师的原始轨迹、指标、日志完全分开；后台进程独立 session，SSH 断联不影响。开发机进程所在主机重启会中断，可用相同命令续跑。GPU worker 更换不影响本机 API 评测。
- 测试成绩仅在完整完成且无未解决基础设施错误时导出正式结果；运行中数字只是进度。

## 文件组织与查看

| 文件 | 用途 |
|---|---|
| `next/outputs/teacher-evaluation/<sol或astra>/progress.md` | 简明实时进度 |
| 同目录 `status.json` / `run.log` / `test.log` | 调度状态、监督日志、逐题进度 |
| 同目录 `test/run.json` / `test/raw.jsonl` / `test/summary.json` | 冻结身份、原始轨迹、完整分母与指标 |
| `next/results/teacher-evaluation/<教师>/summary.json` | **完成后自动生成**：完整结果、配置及原始证据哈希 |
| 本目录 `results/<教师>.md` | **完成后自动生成**：该教师简明正式结果 |

已冻结的公共执行清单保持不变；独立评测入口文件 SHA-256 额外写入 `test/run.json`。后续阶段整理时按当前分支交付正式结果，不提交实时日志。

## 执行与续跑

在现有 Python 运行环境执行，凭证文件放仓库外，通过 `MODELHUB_AK_FILE` 传入。以下路径相对仓库根目录：

```bash
# Sol；对应的私有凭证须已设置。
python next/scripts/teacher_evaluation/run.py --teacher sol \
  --after-collection next/outputs/teacher-parallel/sol \
  --output next/outputs/teacher-evaluation/sol --environment-port 5703

# Astra；使用它自己的私有凭证，采集未完成时只等待。
python next/scripts/teacher_evaluation/run.py --teacher astra \
  --after-collection next/outputs/teacher-parallel/astra \
  --output next/outputs/teacher-evaluation/astra --environment-port 5704
```

实际已通过独立后台 session 启动，不要在它们运行时再次执行命令。相同输出目录有锁；续跑要求原评测入口及共享执行清单不变。
