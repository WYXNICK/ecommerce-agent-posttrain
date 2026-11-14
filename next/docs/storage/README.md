# 存储与恢复

## 保留原则

- 本地保留当前使用的环境、关键 SFT 基座、精确评测用的 LoRA 检查点和原始日志/轨迹；模型和缓存使用开发机 `/tmp`，不放系统盘。
- 非必要合并导出、未选中的旧 SFT 导出，确认 HDFS 成品清单与本地哈希一致后只保留远端副本。`/tmp` 不是长期备份，恢复时核对清单 SHA256。
- pip 下载缓存和未使用的旧编辑器服务端版本可清理；保留活动版本、设置和会话。不自动删除其他项目数据或正在写入的目录。
- 当前可用归档根目录为 `hdfs://harunava/home/byte_data_tt_poi/user/wangyuxuan/shopping-grpo-longhorizon/`。新 feed_quality 路径此前存在读取限制，不将其当作已验证可恢复的备份。

## 已归档的历史 SFT

外层 `outputs/models/` 的大权重和优化器状态已转为 HDFS 保存，配置、训练摘要和 tokenizer 仍在本地；后续如需运行该历史模型，应先恢复。

归档：`hdfs://harunava/home/byte_data_tt_poi/user/wangyuxuan/shopping-grpo-longhorizon/storage/legacy-sft-20261114/models.tar`；同目录 `files.json` 保存每个文件的大小与 SHA256。上传后已流式读回整包并验证哈希。

恢复到开发机 `/tmp`，避免再次占用系统盘：

```bash
mkdir -p /tmp/shopping-legacy-sft
hdfs dfs -get hdfs://harunava/home/byte_data_tt_poi/user/wangyuxuan/shopping-grpo-longhorizon/storage/legacy-sft-20261114/models.tar /tmp/shopping-legacy-sft/models.tar
sha256sum /tmp/shopping-legacy-sft/models.tar
# 与下方保留清单中的 archive_sha256 核对后解包。
tar -xf /tmp/shopping-legacy-sft/models.tar -C /tmp/shopping-legacy-sft
```

恢复后的模型位于 `/tmp/shopping-legacy-sft/models/`，使用时显式指定该路径。

## 已释放的新版副本

本地移除了 9 个已归档导出，共约 37.3 GiB；相应 HDFS 目录、文件清单、关键 base＋adapter 及日志轨迹均保留。清单中的本地路径是原位置，不能再当作现存副本。

具体目录与恢复来源见 [保留清单](../../results/storage/retention-20261114.json)。下载所需远端目录后，按 `artifact-manifest.json` 校验文件。

本次同时清理 pip 缓存及未使用的编辑器旧版本。系统盘可用空间从约 0.5 GiB 增至 17 GiB，`/tmp` 从约 127 GiB 增至 164 GiB。其他项目的大型数据仍占空间，本次未将其当作无用内容清除。
