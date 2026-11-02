# Shopping Agent 新版实现

本目录承载完善版本，流程仍为 **Baseline → SFT → GRPO → Evaluation**。外层保留开源项目、既有实现与历史结果；需要适配的模块先复制到本目录再修改。

## 当前内容

| 路径 | 用途 |
|---|---|
| [docs/project-completion-plan.md](docs/project-completion-plan.md) | **先读此页**：整体步骤、每步作用、当前进度与实验对照 |
| [docs/data-preparation/](docs/data-preparation/README.md) | 数据准备阶段：方案、结果与隔离边界 |
| [docs/baseline/](docs/baseline/README.md) | Baseline 阶段：固定协议、完整成绩与后续比较方式 |
| [docs/teacher-collection/](docs/teacher-collection/README.md) | 教师采集阶段：Sol / Astra 各 2,000＋200 条成品、成功率口径与验收 |
| [docs/sft/](docs/sft/README.md) | SFT 阶段：Astra 2,000＋200、双卡 LoRA、开发集选模与归档 |
| [docs/teacher-evaluation/](docs/teacher-evaluation/README.md) | 教师正式评测：Sol 80.4%、Astra 80.9%，各完整 1,000 题 |
| [docs/data-preparation/methodology.md](docs/data-preparation/methodology.md) | 数据筛选规则、分布、复现与详细使用说明 |
| [docs/execution.md](docs/execution.md) | 统一入口、冻结协议、教师 API 与运行命令 |
| [docs/resource-usage.md](docs/resource-usage.md) | 两教师采集及评测的 token、缓存明细与费用核算状态 |
| [configs/data_selection.json](configs/data_selection.json) | 正式 Astra 数据与独立 Sol 参考数据的选择，清单路径相对 `next/` |
| [results/baseline/](results/baseline/summary.json)、[results/teacher-collection/sol/](results/teacher-collection/sol/summary.json) | 可核验的阶段指标与验收记录 |
| [Sol 数据](data/teacher/sol/manifest.json)、[Astra 数据](data/teacher/astra/manifest.json) | 已冻结的压缩 SFT 成品及 SHA-256 |
| [configs/execution.json](configs/execution.json) | 学生、教师和执行预算的权威配置 |
| [scripts/agent.py](scripts/agent.py) | 校验、采集、重建和学生评测入口 |
| [configs/task_pool.json](configs/task_pool.json) | 冻结的任务准备配置 |
| [scripts/prepare_task_pool.py](scripts/prepare_task_pool.py) | 离线质量筛选、分组去重与分层划分 |
| [data/task_pool/metadata.json](data/task_pool/metadata.json) | 数据分布、来源和 SHA-256 校验清单 |
| [tests/test_prepare_task_pool.py](tests/test_prepare_task_pool.py) | 划分、哈希、奖励与去重验收 |

已冻结教师候选 4,000、GRPO 2,000、SFT 验证候选 800、开发 300、测试 1,000，另有训练备用 3,977。Baseline、Sol / Astra 各 2,000＋200 条教师数据及各完整 1,000 题教师评测均已完成，并通过全量离线复核。**正式教师及 SFT 数据采用 Astra；Sol 原样保留为独立参考。** 新版 SFT 已在双 H100 启动正式训练，详见 SFT 阶段说明；GRPO 尚未执行。

每阶段文档在 `docs/<阶段>/`，指标在 `results/<阶段>/`，数据成品在 `data/`。自动更新的运行报告与完整日志保留本地，采集与评测分开：Astra 采集报告在 `next/docs/teacher-collection/runtime/astra-current.md`，教师评测在 `next/outputs/teacher-evaluation/<教师>/progress.md`。每个阶段完成后整理并推送当前分支，规则见 [AGENTS.md](AGENTS.md)。

## 只读复用与路径约定

- 环境代码：外层 `environments/ShopSimulator/shop_env/`。
- 商品库、搜索索引：路径由本目录 `configs/task_pool.json` 指定，相对于仓库根目录。
- 环境契约：外层 `data/environment.json`；原 Final-200：外层 `data/evaluation/tasks.jsonl`。
- `metadata.json` 的运行时哈希路径、历史来源路径均相对于仓库根目录；成品哈希路径相对于 `next/data/task_pool/`。
- 新版配置和输出默认定位到本目录，不依赖启动时的工作目录。历史证据见外层 [docs/reproduction/README.md](../docs/reproduction/README.md)。

## 离线复现

以下命令在仓库根目录执行，使用现有环境依赖：

```bash
environments/ShopSimulator/.venv-shopsim/bin/python next/scripts/prepare_task_pool.py
environments/ShopSimulator/.venv-shopsim/bin/python -m unittest discover -s next/tests -p test_prepare_task_pool.py -v
```

默认复用已冻结的历史用题快照。完整采集与学生初始评测的命令见[执行说明](docs/execution.md)；不要为了复盘重复启动已完成阶段。
