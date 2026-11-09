# 当前可交付结果与候选保存

## 展示口径

主线为 **Baseline → S0 → 修复 SFT → 原终局 GRPO step500**。原终局 step400 是测试前固定的次候选；S0 直接 GRPO 为起点消融，局部信用与稳定终局继续作为探索备份。均保留，不把不同起点或不同协议的结果混为同一增益。

| 模型 | 用途 | test1000 严格成功率 | Reward v3 |
|---|---|---:|---:|
| 原始学生 | Baseline | 1.5% | −0.1039 |
| Astra | 教师参考（推理预算不同） | 80.9% | 0.7837 |
| S0 step750 | 初始 SFT / 消融起点 | 75.0% | 0.7034 |
| 修复 SFT step72 | 原终局 GRPO 起点 | 76.6% | 0.7229 |
| **原终局 GRPO step500** | **当前主结果，开发集选中** | **77.8%** | **0.7364** |
| 原终局 GRPO step400 | 固定 Reward 次候选 | 77.0% | 0.7359 |
| S0 GRPO step450 | 起点消融，开发集选中 | 75.9% | 0.7156 |
| S0 GRPO step500 | 起点消融，最终检查点 | 75.9% | 0.7148 |

表格来自两份权威测试汇总：[原六模型](../../results/evaluation/test-comparison/summary.json)、[S0 GRPO 对照](../../results/evaluation/s0-grpo-test/summary.json)。逐题差异、失败类型、动作数与统计限制见[原结果](results.md)、[S0 消融结果](../grpo-s0/test-results.md)。

**可用结论：** 原终局 GRPO 相对其修复 SFT 起点提高 1.2 个百分点；S0 直接 GRPO 相对 S0 提高 0.9 个百分点。两者均为单训练种子，配对检验未达到常用 0.05 阈值，不表述为已证明稳定或显著提升。S0 到主结果的 2.8 个百分点包含修复 SFT 和 RL 的共同贡献。测试结果不能用于重新选检查点、针对失败题训练或事后挑选指标。

## 保存与恢复

[候选清单](../../results/evaluation/final-candidates/manifest.json) 是模型身份、用途及归档证据的索引。保留 S0/修复 SFT 基座及所选 SFT 检查点、四个 GRPO adapter 检查点、训练日志/轨迹和两轮 test1000 完整证据。现有其他检查点、导出副本和历史结果保持原样。

- **精确复现评测：** 使用冻结配置指定的基座＋原始 adapter，验证对应 SHA256；不要用 BF16 合并副本代替评测策略。SFT 为已合并基座，教师为 API 模型，没有可归档教师权重。
- **本地副本：** 开发机 `/tmp/shopping-next-models/20261104/`，不依赖 Worker；`/tmp` 仍属于临时存储。
- **已完成 HDFS 备份：** `hdfs://harunava/home/byte_data_tt_poi/user/wangyuxuan/shopping-grpo-longhorizon/next/20261104/`，保持相对目录；成品清单保存文件哈希与大小。
- **后续备用 HDFS：** `hdfs://harunava/home/byte_tiktok_feed_quality/wangyuxuan/shopping-grpo-longhorizon/next/`。用户已授权使用；当前访问返回 `GDPR permission denied / DirectAccessControl`，尚未在此完成备份。权限可用后按运行标识分目录归档；切换时使用独立目标回执，不能把旧目录成功回执当作新目录成功。

仓库保存实现、冻结配置和结果摘要，大权重与完整日志留在归档。既有备份不删除；当前归档服务继续写已验证可用的旧 HDFS，避免因新目录权限问题中断保护。
