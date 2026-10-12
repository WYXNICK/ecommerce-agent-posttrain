# Shopping Agent 新版实现

本目录承载完善版本，流程仍为 **Baseline → SFT → GRPO → Evaluation**。外层保留开源项目、既有实现与历史结果；需要适配的模块先复制到本目录再修改。

## 当前内容

| 路径 | 用途 |
|---|---|
| [docs/project-completion-plan.md](docs/project-completion-plan.md) | 整体目标、实施阶段与对照设计 |
| [docs/data-preparation.md](docs/data-preparation.md) | 已完成的数据筛选、数量、隔离与使用说明 |
| [configs/task_pool.json](configs/task_pool.json) | 冻结的任务准备配置 |
| [scripts/prepare_task_pool.py](scripts/prepare_task_pool.py) | 离线质量筛选、分组去重与分层划分 |
| [data/task_pool/metadata.json](data/task_pool/metadata.json) | 数据分布、来源和 SHA-256 校验清单 |
| [tests/test_prepare_task_pool.py](tests/test_prepare_task_pool.py) | 划分、哈希、奖励与去重验收 |

本轮迁移保持全部 12 个数据及审计文件内容不变，仅更新代码定位和清单中的实现路径/哈希。已冻结教师候选 4,000、GRPO 2,000、SFT 验证候选 800、开发 300、测试 1,000，另有训练备用 3,977；仍处于任务准备阶段，未生成新版教师轨迹。

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

默认复用已冻结的历史用题快照。原有采集、训练和评测入口尚未复制或适配到本目录，后续按完善方案逐项实施。
