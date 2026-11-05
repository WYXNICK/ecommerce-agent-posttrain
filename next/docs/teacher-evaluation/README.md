# 教师评测

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

## Astra 开发集评测

**状态：已完成。** Astra 完整 dev300 严格成功 **245/300（81.67%）**，与本轮已完成的 S0（220/300）、普通增量组（218/300）、修复组（226/300）做同题能力与失败对照。复用现有学生评测，不重复训练或运行学生测试集。

沿用同一 Astra 快照、high 推理、6 路并发、共享环境与可见预算；每题一次有效行为，模型失败不重采。开发集结果用于诊断，不作为独立测试成绩或教师能力上限。逐题对照须核对任务 ID、执行指纹与完整分母，教师与学生推理成本不同。

结果独立存放：`outputs/teacher-evaluation/astra-dev/dev/` 为原始证据，`results/teacher-evaluation/astra/dev/` 为正式汇总；不覆盖下方已完成的 test1000 成绩。 完成后使用 `scripts/analysis/teacher_dev.py` 校验完整分母与协议，生成与三个 SFT 候选的逐题配对分析；自动同步阶段报告与总计划，并备份证据、推送当前分支。结果及局限见[开发集报告](results/astra-dev.md)汇总结果与师生得失。

```bash
python next/scripts/teacher_evaluation/run.py --teacher astra --split dev \
  --after-collection next/outputs/teacher-parallel/astra \
  --output next/outputs/teacher-evaluation/astra-dev --environment-port 5707
```

## 已完成的 test1000 结果

| 教师 | 完成 | 严格成功 | 成功率 | 平均 Reward v3 | 平均步骤 |
|---|---:|---:|---:|---:|---:|
| Sol | 1,000/1,000 | 804 | **80.4%** | 0.7786 | 11.129 |
| Astra | 1,000/1,000 | 809 | **80.9%** | 0.7837 | 10.390 |

Sol 最终汇总于北京时间 **2026-10-30 02:05** 落盘，Astra 于 **05:59** 落盘。Astra 第一条评测 episode 创建于 **03:54:42**，晚于完整采集结束 **03:54:22**，自动衔接按预期执行。监督状态、原始轨迹、正式导出指标及哈希已全量核对，两组均无缺题、无未解决基础设施错误，每题只有一次有效模型行为。

### 如何解读

- **能力接近，未证明明显优劣。** 779 题共同成功，Astra 独有成功 30 题、Sol 独有成功 25 题，166 题均失败。净差 0.5 个百分点；任务层配对 bootstrap 95% 区间为 **−1.0 至 +1.9 个百分点**，McNemar 精确双侧 p≈0.59。只有每模型一次执行，没有估计重复调用波动。
- **Astra 本次执行更省步骤，但存在不同取舍。** response 超预算 36 题，少于 Sol 的 66；部分满足购买 104 题，多于 Sol 的 77；错误购买均为 8。不能把所有未成功都归因于服务问题。
- 已报告总 API token：Sol **79,070,330**、Astra **70,945,327**。这是包含重复 prompt 的 usage，不能当成实际账单或直接推导费用比例。
- 采集与正式评测的统一 token、缓存明细及费用状态见[用量说明](../resource-usage.md)；实际 ModelHub 金额尚待计费规则与账单核算。
- 测试结果只作为固定能力参考，不用于选择回流题、修改测试或调参。后续学生提升必须来自真实训练，并在同一协议下独立验证。

单模型报告：[Sol](results/sol.md)、[Astra](results/astra.md)。配对计数、方法及全量复核：[comparison.json](../../results/teacher-evaluation/comparison.json)。

## 执行方式与自动衔接

- **Sol：采集完成后直接执行正式 1,000 题，已完成。**
- **Astra：后台等待完整采集完成后自动执行，已完成。** 必须同时满足监督状态 complete、训练 2,000、验证 200 且身份/协议一致；采集失败或部分完成不会触发评测。
- 两个教师的原始轨迹、指标、日志完全分开；后台进程独立 session，SSH 断联不影响。开发机进程所在主机重启会中断，可用相同命令续跑。GPU worker 更换不影响本机 API 评测。
- 测试成绩仅在完整完成且无未解决基础设施错误时导出正式结果；运行中数字只是进度。

## 文件组织与查看

| 文件 | 用途 |
|---|---|
| `next/outputs/teacher-evaluation/<sol或astra>/progress.md` | 简明实时进度 |
| 同目录 `status.json` / `run.log` / `test.log` | 调度状态、监督日志、逐题进度 |
| 同目录 `test/run.json` / `test/raw.jsonl` / `test/summary.json` | 冻结身份、原始轨迹、完整分母与指标 |
| `next/results/teacher-evaluation/<教师>/summary.json` | 已生成：完整结果、配置及原始证据哈希 |
| 本目录 `results/<教师>.md` | 已生成：该教师简明正式结果 |

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

上述 test1000 已完整完成，不需要重复运行。相同输出目录有锁；如处理其他尚未完成的运行，续跑要求原评测入口及共享执行清单不变。
