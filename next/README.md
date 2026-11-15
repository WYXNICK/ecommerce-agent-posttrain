# E-Commerce Agent — 训练与评测实现

本目录为当前主线的完整执行栈：**Baseline → SFT → 纠错 SFT → GRPO → Evaluation**。

| 内容 | 入口 |
|---|---|
| 总体设计与完成状态 | [project-completion-plan.md](docs/project-completion-plan.md) |
| 主结果与统计解释 | [核心结果](../docs/results.md) |
| 当前全部实验 | [实验档案](../docs/experiments.md) |
| 环境与运行 | [运行指南](../docs/reproduce.md)、[执行协议](docs/execution.md) |
| 数据、训练与评测文档 | [阶段导航](../docs/README.md) |
| 模型、日志与恢复 | [工件索引](../docs/artifacts.md) |

`src/` 存共享执行器，`scripts/` 存各阶段入口，`configs/` 存固定配方，`data/` 存成品，`results/` 存机读结果，`docs/<阶段>/` 存方法与总结。大体积原始日志和轨迹放在 `outputs/` 并外部归档，不提交 Git。

环境只读复用仓库的 `environments/ShopSimulator/shop_env/`；任务准备配置中的来源路径相对仓库根目录。内部路径是执行契约的一部分，不因展示名称变化而移动。根目录的早期执行栈与200题结果仅作历史恢复，当前任务请使用本目录入口。
