# 普通增量 SFT 补充评测

**状态：已完成。** 本实验仅作备份对照，不纳入正式主结果。

## 目的与方法

补齐普通增量 SFT 的泛化结果，与已完成的冷启动 SFT、纠错 SFT 对照。固定使用 dev300 选中的 step72（218/300，72.67%），不根据测试结果重新选模。

复用冻结 test1000、Environment v2.1、Reward v3、observation v2、tool schema v2 和相同执行预算。严格成功要求完整且可验证的 gold_purchase。每题执行一次，报告成功率、平均奖励、平均步骤和同题得失；基础设施错误需处理完成后再汇总。

## 记录

- [固定模型与评测配置](../../configs/s_general_test.json)
- 原始逐题轨迹、运行配置与日志保存在 `next/outputs/evaluation/s-general-test/`（不入 Git）。
- [结果报告](general-test-results.md)：749/1,000（74.90%）；[完整汇总](../../results/evaluation/s-general-test/summary.json)。
