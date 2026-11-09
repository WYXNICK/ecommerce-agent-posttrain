# S0 GRPO 测试集对照

**状态：已完成。** 测量从初始 SFT 直接进行终局 GRPO 的泛化收益，与 S0、修复 SFT 和原终局 GRPO 对照。

## 方法

- 固定 step450（开发集选中）和 step500（最终检查点），使用精确 S0 基座＋原始 LoRA；权重哈希见 [冻结清单](../../configs/s0_grpo_test.json)。不依据测试结果重新选模。
- 完整冻结 test1000；沿用 Environment v2.1、Reward v3、观察、Action Guard、预算、temperature=0 与关闭 thinking 的学生协议。每题一次有效执行，模型失败计入分母，基础设施失败需补齐。
- 两张 H100 各服务一个候选，每模型 8 路并发；复用经任务 ID、源码指纹、轨迹与文件哈希核验的 Baseline、Astra、S0、S-repair、原 GRPO step400/500 结果。
- 报告成功率、Reward、动作数、错误购买、循环、token 用量及同题成功/失败变化。配对检验作描述，不声称单种子或重复使用测试集能证明稳定泛化；测试轨迹不回流训练。

## 交付

原始日志和逐题轨迹保存在 `next/outputs/evaluation/s0-grpo-test/`，结束后复制开发机 `/tmp` 并进入 HDFS 归档队列。后台监督器正常或异常退出均恢复双卡保活。结果写入 [测试汇总](test-results.md)，机器可读汇总位于 `next/results/evaluation/s0-grpo-test/`；原六模型报告保持不变。
