# 图表与来源

首页呈现系统、主结果与真实案例；数据划分和检查点曲线放在对应阶段文档。所有新图使用统一的 STIX 字体、蓝／金／绿配色和矢量输出；颜色之外同时用线型、标记和文字区分含义。

| 图 | 内容与依据 | 下载 |
|---|---|---|
| 系统架构 | [共享执行协议与模块职责](../architecture.md)，区分训练流程、执行循环和学习信号 | [SVG](system-architecture.svg) · [PDF](system-architecture.pdf) · [PNG](system-architecture.png) |
| 数据筛选与划分 | [冻结任务清单](../../next/data/task_pool/metadata.json)，展示筛选数量与互斥集合；子集不重复计数 | [SVG](data-partitions.svg) · [PDF](data-partitions.pdf) · [PNG](data-partitions.png) |
| 真实纠错案例 | [任务 19141 原始证据摘录](../../next/results/data-reflow/r1/case-19141.json)，教师接管与后缀监督 | [SVG](correction-case.svg) · [PDF](correction-case.pdf) · [PNG](correction-case.png) |
| GRPO 检查点 | [同一运行的 dev300 汇总](../../next/results/grpo/outcome/summary.json)，每 50 次更新评测，保留未经平滑的数值 | [SVG](grpo-checkpoints.svg) · [PDF](grpo-checkpoints.pdf) · [PNG](grpo-checkpoints.png) |
| 固定测试结果 | [test1000 汇总](../../next/results/evaluation/test-comparison/summary.json)，相同任务与执行预算 | [SVG](test-results.svg) · [PDF](test-results.pdf) · [PNG](test-results.png) |
| 多次采样诊断 | [dev300 pass@k 汇总](../../next/results/passk/dev300/summary.json)，任务配对 bootstrap 区间 | [SVG](sampling-results.svg) · [PDF](sampling-results.pdf) · [PNG](sampling-results.png) |

## 生成

依赖 Python 与 Matplotlib，在仓库根目录执行：

```bash
python docs/assets/render_results.py
python docs/assets/render_methods.py
```

输出 SVG、嵌入字体的 PDF、300 DPI PNG。统计图直接读取归档汇总；案例图读取证据摘录，英文文字为原始中文的简写。架构图表达当前主线设计，不代表探索方案均已进入主结果。

## 视觉参考

- [Agent Lightning](https://arxiv.org/abs/2508.03680)：训练与执行分层，使用明确的模块边界。
- [veRL HybridFlow](https://verl.readthedocs.io/en/latest/hybrid_flow.html)：区分算法流程与执行职责。
- [Building effective agents](https://www.anthropic.com/research/building-effective-agents)：简洁呈现模型、工具与环境反馈循环。
- [ICML 论文模板](https://media.icml.cc/Conferences/ICML2025/Styles/example_paper.pdf)：紧凑多面板、图注外置及适合缩放的矢量图。

这些来源用于图示组织参考；本目录六张方法与结果图按项目实现和证据重新绘制。ShopSimulator 环境示意图的来源另见[依赖与来源](../../THIRD_PARTY_NOTICES.md)。
