# 新版执行入口

本页是采集、评测与共享执行循环的操作参考。整体顺序见[项目计划](project-completion-plan.md)，已完成结果见 [Baseline 总结](baseline/README.md)与[教师采集总结](teacher-collection/README.md)。新版 SFT 执行入口见 [SFT 阶段说明](sft/README.md)；以下命令用于复现或续跑，不代表需要重复执行已完成阶段。

## 1. 冻结协议

权威配置：[execution.json](../configs/execution.json)。源码、配置、参考环境和依赖版本的校验清单：[execution_manifest.json](../configs/execution_manifest.json)。修改后必须审阅并重新冻结，旧运行目录不能混入新协议。

| 项目 | 统一规则 |
|---|---|
| 环境/计分 | Environment v2.1、Reward v3、Observation v2、Tools v2 |
| 学生 | Qwen3.5-2B，固定 revision，关闭 thinking |
| 执行 | 每轮最多执行一个工具；动作只取当前页面合法目标；连续 3 次非法动作结束 |
| 步数 | 最多 35 个执行步骤；有效终局优先于下一轮预算检查 |
| Token | 学生 tokenizer：上下文 24,576；累计 response 20,480；每轮 assistant ≤512；上下文预留 512 |
| 页面 | 搜索 1,536、详情 4,096、其他 768 token；保留可执行目标与必要页尾 |
| 严格成功 | 完整终局，`gold_purchase`、`reward_valid=true`、`purchase_success=true` |

累计 response 包含初始提示之后的 assistant 和 tool 上下文，以学生模板实际渲染长度计数；教师私有推理 token 单独记录，不进入学生监督。未来 SFT/GRPO 必须复用该协议及预算，不能直接把外层旧训练入口当成已经适配的新入口。

共享实现：[runtime.py](../src/shopping_agent/runtime.py)。失败、提前结束和超预算保留原始轨迹；基础设施异常停止新增任务，释放在途租约并保存证据，修复后原命令续跑。SIGINT/SIGTERM 停止新增任务，等待在途 episode 结束后保存；SIGKILL 不保证释放租约。

## 2. 教师接口

当前并行使用 **gpt-6-astra**（`gpt-6-astra-2026-09-03`）与 **GPT-6.1-Sol**（`gpt-6.1-sol-2026-09-29`），均为 `high` 推理。默认配置选择 Astra；`--teacher-profile sol` 只覆盖教师身份，共享冻结的学生与环境协议。快照变化会停止采集，两个教师的数据独立计数、验收与保存。

用户提供的 ModelHub `crawl` 网关不支持 Sol 的“原生 function tools＋推理”组合；Sol 和 Astra 均已验证可以采用 **严格 JSON Schema 动作输出＋真实工具历史**：

1. 从唯一工具定义生成动作 schema，教师只选择工具名和参数。
2. 转成统一 `assistant.tool_calls`，由同一个 Action Guard 验证并执行真实环境。
3. 真实 observation 作为 `tool` 消息返回教师，持续执行到终局。

这属于结构化动作适配，不能称为该网关的原生 function calling。不会编造工具结果，不需要教师 logits、tokenizer 或私有思维链。教师接口单次输出上限 8,192，包含私有推理；可见动作仍受学生侧每轮 512 的预算约束。

AK 只从 `MODELHUB_AK` 或 `MODELHUB_AK_FILE` 读取，不写入配置、日志或数据。文件格式见 [.env.example](../.env.example)。usage、推理 token、模型快照、请求 ID 和延迟保留在原始轨迹；未返回 usage 的重试费用不能伪称已完整统计。

## 3. 使用流程

在已经配置的 Python 运行环境中，从仓库根目录执行。环境和缓存放 `/tmp`；已有 `HF_HOME` 等设置优先保留。

```bash
# 校验冻结数据、源码、依赖与学生 tokenizer，不调用模型 API。
python next/scripts/agent.py check

# 独立环境服务；默认端口 5701、8 个租约槽，日志落在 /tmp。
bash next/scripts/start_environment.sh
```

随后启动完整采集，训练和验证独立运行：

```bash
export MODELHUB_AK_FILE=/tmp/private-modelhub.json
python next/scripts/agent.py collect --role train \
  --output next/outputs/teacher-train
python next/scripts/agent.py collect --role validation \
  --output next/outputs/teacher-validation
```

- 默认目标：**2,000 个唯一可训练任务＋200 个独立验证任务**；各使用冻结的 4,000/800 候选池。
- 留出校验自动强制执行；不能用参数关闭。验证任务不进入训练，开发/测试及历史重复组均受保护。
- 按候选池九领域占比分配验收配额，每题最多两次；先遍历未尝试任务，再重试模型失败。并发不会超收同一任务或配额。
- 严格成功后，再做学生模板、完整序列长度和监督 mask 检查；无截断凑数。配额不足会报告缺口、非零退出，不擅自换题或扩大候选池。
- 相同命令、相同输出目录自动续采；基础设施失败可重试，不耗模型行为尝试次数。可增加 `--build-only`，仅从原始证据重建成品。
- `--subset` 只能选当前角色候选池内的唯一任务；属于显式子集运行，不能冒充完整采集或正式测试。

学生评测：

```bash
# CHECKPOINT 是已导出的真实权重目录；此命令会使用 GPU。
bash next/scripts/serve_model.sh "$CHECKPOINT"
python next/scripts/agent.py evaluate --split dev --checkpoint "$CHECKPOINT" \
  --output next/outputs/baseline-dev
# 方法和候选冻结后，才运行正式 1,000 题；所有模型各自运行同一测试集。
python next/scripts/agent.py evaluate --split test --checkpoint "$CHECKPOINT" \
  --output next/outputs/baseline-test
```

评测先核对 vLLM 实际服务的模型路径并记录权重哈希。固定全部预期任务为成功率分母；基础设施未解决或任务缺失时，结果标记未完成，不能报告为正式完整结果。300-dev 用于开发、选模；最终 Baseline/SFT/GRPO 都要在同一 1,000-test 上比较。

正式阶段由 `scripts/run_stage.py` 监督：拥有独立环境/模型服务，分开保存各任务日志，每 10 秒更新简明任务报告；完成或异常后清理所拥有的服务，Baseline 同时恢复已有双卡保活。用 `nohup`/独立 session 启动后，SSH 断联不会终止任务；worker 回收、机器重启仍会中断进程，须按冻结配置续跑。

本轮 execution v1.3 为**每教师 6 路并发**，各自拥有 8 槽环境服务；Astra 端口 5701，Sol 端口 5702。Sol 使用 `--teacher-profile sol`；凭证分别通过私有环境变量传入。任务划分、学生、奖励、守卫、教师推理参数及 token 预算未变，Baseline 无需重跑。

**429 每请求最多尝试 12 次**，等待 **2/4/8/16 秒，之后每次 20 秒**，十一段等待合计 170 秒，另计请求本身耗时。其他临时传输错误最多 **3 次尝试**、等待 **5/10 秒**，鉴权失败立即停止。重试不重复执行环境动作；耗尽后保存进度停止，不无限重启整批采集。更高并发不保证更快，仍受 API 容量约束。

`--seed-from` 让新阶段只读继承同教师、同角色、同候选与配额的既有原始证据，拒绝混用教师或修改候选；`source.json` 锁定来源 run/raw 的 SHA-256。每次启动重新按学生模板验收，成品和统计包含旧证据及新轨迹；新目录的 `raw.jsonl` **仅含本阶段新增轨迹**，完整来源需结合 `source.json` 查看。两个教师均补齐到各自 2,000＋200 条，验证已足额时不重复调用 API。

本批轨迹采完后可反复用于 SFT，训练无需重复调用教师；后续两轮失败回流仍需教师生成修复轨迹。学生 SFT 参数优化、GRPO rollout 和局部信用分支不调用教师；评测裁判另行配置，不等于后续完全不需要 API。

## 4. 每次运行只看这些文件

| 文件 | 用途 |
|---|---|
| `run.json` | 不可混用的协议、候选、教师或模型权重身份 |
| `raw.jsonl` | 追加并落盘的原始证据，续跑和重建的权威来源 |
| `source.json` | 续采时只读继承的旧证据目录及哈希；不存在时仅使用本目录 raw |
| `summary.json` | 当前状态、数量/缺口、失败和 usage；评测另含指标 |
| `train.jsonl` / `validation.jsonl` | 去重后的可训练成品，仅采集生成 |
| `rejected.jsonl` | 每次未进入成品的原因，仅采集生成 |

命令标准输出每完成一个任务打印一行进度，可自行重定向为一个 `run.log`。同一目录有进程锁，禁止重复启动。不要手工修改 `raw.jsonl`；衍生文件损坏可由原始证据重建。

## 5. 验收与归档

`python -m unittest discover -s next/tests -v` 覆盖守卫、预算、失败分类、配额去重、模板、留出保护、脱敏、固定评测分母、短间隔有界重试和继承证据的隔离。

完整结果归纳到 `docs/<阶段>/`，指标及成品哈希按阶段发布到 `results/` 和 `data/`。现有 `next/outputs/teacher-parallel/` 保留原始采集、继承清单与全量验收证据；`next/outputs/baseline-v2/` 保留 Baseline 逐题证据，不因整理目录改动原始证据。新教师评测统一放 `next/outputs/teacher-evaluation/<教师>/`，方法见[教师评测说明](teacher-evaluation/README.md)。自动运行报告在本地更新，不替代阶段总结。

仅在确实修改协议并审阅后使用 `python next/scripts/agent.py freeze`；它不会验证修改合理性，也不能用来绕过已开始实验的冻结约束。
