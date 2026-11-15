# 实验档案

主展示只采用 [S0 → 纠错 SFT → 终局 GRPO](results.md)。本页保留所有主要尝试，包括无收益、暂停和历史协议；“保留起点”不记为新方法提升。dev 与 test 均为严格成功题数，分母分别为300、1,000。

| 实验 | 问题与主要变化 | 结果与定位 | 方法 / 结果 |
|---|---|---|---|
| 冷启动 SFT | 2,000条教师示范，动作监督 | dev220，test750；主线 | [SFT](../next/docs/sft/README.md) |
| 普通增量 SFT | 144条普通新增示范＋144回放 | dev218，低于S0；对照 | [回流](../next/docs/data-reflow/results.md) |
| 纠错 SFT | 144条失败后缀纠错＋144回放 | dev226，test766；主线 | [回流](../next/docs/data-reflow/README.md) |
| 终局 GRPO，2×4 | 原始终局奖励＋模型失败学习 | step500 dev228、test778；主结果；step400 test770作为次候选保留 | [方法](../next/docs/grpo/README.md) / [结果](../next/docs/grpo/results.md) |
| S0直接GRPO | 只改变RL起点 | step450 dev226；450/500 test均759；起点消融 | [方法](../next/docs/grpo-s0/README.md) / [测试](../next/docs/grpo-s0/test-results.md) |
| GRPO，4×4 | 增大有效任务batch，仍每题4条 | 训练检查点最高dev225，保留起点226；无test | [方法](../next/docs/grpo-batch/README.md) / [结果](../next/docs/grpo-batch/results.md) |
| 局部信用GRPO | 在局部状态追加分支，改进信用分配 | 最高dev225，保留起点226；探索 | [方法](../next/docs/grpo-local/README.md) / [结果](../next/docs/grpo-local/results.md) |
| 稳定终局GRPO | 8×4、KL约束等稳定化配置 | step225 dev227，最终224；未证明优于主结果 | [方法](../next/docs/grpo-stable/README.md) / [结果](../next/docs/grpo-stable/results.md) |
| SDPO R1 | 成功轨迹/环境反馈构造自教师 | step150 dev229、test756；后期退步 | [方法](../next/docs/sdpo/r1-method.md) / [测试](../next/docs/sdpo/test-results.md) |
| SDPO R2 | 失败反馈EMA纠正＋冻结SFT成功保护 | 保留起点dev226，无test；探索 | [方法](../next/docs/sdpo/README.md) / [结果](../next/docs/sdpo/r2-results.md) |
| 证据约束终局GRPO | 明确必选条件违反不能被部分匹配抵消 | step500 dev230、test776；step450 test776，均未超过主结果 | [设计](../next/docs/grpo-constraints/README.md) / [测试](../next/docs/grpo-constraints/test-results.md) / [450](../next/docs/grpo-constraints/test-450-results.md) |
| 严格成功优先GRPO | 有界失败奖励，仅更新成功/失败混合组 | step50 dev227、test765；未获整体收益 | [设计](../next/docs/grpo-success/README.md) / [测试](../next/docs/grpo-success/test-results.md) |
| GRPO后纠错SFT | 定向纠错＋成功回放 | 采集暂停，未训练；备选，不属主线 | [方案](../next/docs/consolidation/README.md) |
| Hard Set | 结构难点与可执行性筛选 | 105待验收候选，未作为正式测试 | [标准与审计](../next/docs/hard-evaluation/README.md) |
| 四模型pass@k | 相同题目与温度下测量单次可靠性/覆盖 | 9,600条执行记录，已完成 | [方案](../next/docs/passk/README.md) / [结果](../next/docs/passk/results.md) |
| Sol独立教师 | 不同教师的采集及参考评测 | 2,000＋200条成品，test804；不混入Astra训练 | [采集](../next/docs/teacher-collection/README.md) / [评测](../next/docs/teacher-evaluation/README.md) |

## 从探索中得到什么

- **更多batch或更复杂目标没有自动带来更好结果。** 原2×4终局方案保留为主结果；4×4、KL与局部信用只报告真实结果，不从失败实验中反推确定因果。
- **奖励正确性与优化效果不同。** 证据约束修复了有反证却仍获部分匹配分的问题，但test成功率未超过主结果；成功优先虽限制全失败组的正向强化，也减少可用训练组，最终仍无净增益。
- **稳定化不是提升保证。** SDPO R2比R1部分同期检查点退化更少，但最终保留SFT起点；不能写成“自蒸馏带来提升”。
- **解码协议影响结论。** 纠错SFT提高贪心成功率却降低随机pass@1；主GRPO的可靠性收益由独立多次采样诊断支持。

这些对照大多为单种子，部分同时改变多个因素；不把相关性写成组件级因果证明。继续研究时以训练池证据形成假设，独立运行、dev选模，再冻结候选测试。

## 历史与完整记录

- [早期200题实验：SFT、GRPO、E1、E2、E3](reproduction/README.md)，使用独立协议，不能与test1000相减。
- [历史运行身份](../next/results/history/20261103/index.json)，保留旧数据与模型来源，不冒充当前模型的新采样。
- [性能试验](../next/results/grpo/batch-performance/summary.json)，与训练质量结果分开保存。
- [工件索引](artifacts.md)：配置、源码快照、训练日志、轨迹、检查点与归档回执。
