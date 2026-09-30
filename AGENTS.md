# Repository contract

This repository supports one workflow only:

```text
Baseline → SFT → GRPO → Evaluation
```

The runtime contract is ShopSimulator Environment v2.1, Reward v3, observation
v2 and tool schema v2. Do not add compatibility launchers, historical datasets,
old benchmarks, machine-specific paths or experiment journals.

Training data must never overlap `data/evaluation/tasks.jsonl`. Strict success
requires a complete `gold_purchase` terminal result with `reward_valid=true`.

Do not start training, merge models or run the 200-task evaluation unless the
user explicitly requests execution.

## 开发原则

后续修改代码或新增实现时，必须遵循以下原则：

- **简单且正确。** 尽量少的代码和组成部分，同时保持清晰可读。
- **只处理真实可能发生的失败。** 外部输入在边界校验；内部信任已建立的契约，不为不可能状态堆防御分支、兜底和恢复逻辑。
- **小而内聚的函数。** 职责、输入输出、依赖明确；按职责拆分，不为缩短行数机械拆函数。
- **偏好纯函数和不可变数据。** 副作用集中在明确边界，依赖显式传入，避免隐藏修改和共享可变状态。
- **重复或结构不合适时先重构。** 减少特殊分支、包装层和补丁；抽象围绕真实领域概念或已存在的共性，不预建通用框架。
- **每个事实只保留一个权威来源。** 能从规范状态推导的数据，就不再维护一份需要同步的状态。
- **防御要适当，避免过度防御逻辑。** 对非实际可能发生的场景，不单独添加防御逻辑。
