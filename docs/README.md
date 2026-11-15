# 文档导航

## 首次阅读

1. [项目首页](../README.md)：任务、贡献与主结果。
2. [整体设计](../next/docs/project-completion-plan.md)：每阶段做什么、为何做、当前完成状态。
3. [架构](architecture.md)：数据、执行、训练与评测如何连接。
4. [核心结果](results.md)：test1000、pass@1/pass@8、配对差异及结论边界。
5. [运行指南](reproduce.md)：环境校验、训练评测与继续修改 Reward。

## 阶段方法与结果

| 阶段 | 方法与结果入口 |
|---|---|
| 数据筛选与隔离 | [数据准备](../next/docs/data-preparation/README.md) |
| 领域 Baseline | [初始评测](../next/docs/baseline/README.md) |
| 教师示范与教师参考成绩 | [采集](../next/docs/teacher-collection/README.md)、[评测](../next/docs/teacher-evaluation/README.md) |
| 冷启动 SFT | [方法](../next/docs/sft/README.md)、[结果](../next/docs/sft/results.md) |
| 失败回流与纠错 SFT | [方法](../next/docs/data-reflow/README.md)、[结果](../next/docs/data-reflow/results.md)、[真实案例](../next/docs/data-reflow/case-study.md) |
| 终局 GRPO | [方法](../next/docs/grpo/README.md)、[训练结果](../next/docs/grpo/results.md) |
| 测试与多次采样诊断 | [统一测试](../next/docs/evaluation/results.md)、[pass@k](../next/docs/passk/results.md) |

## 复盘与维护

- [图表索引](assets/README.md)：架构、数据划分、纠错案例与结果图的来源、下载和复现。

- [实验档案](experiments.md)：全部主线、消融、未增益方案和暂停项目。
- [工件与日志](artifacts.md)：机读结果、训练证据、归档回执与恢复方式。
- [教师资源统计](../next/docs/resource-usage.md)：token、缓存及尚不能确定的费用。
- [早期 200 题实验](reproduction/README.md)：独立历史协议，不与当前 test1000 混算。
- [早期设计资料](archive/README.md)：历史实现与研究记录，只供复盘。

主结果以机读阶段结果为依据；新实验使用独立输出目录。阶段总结说明目的、方法、结果和边界，逐步日志保存在运行工件中。
