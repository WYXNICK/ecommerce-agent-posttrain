# E-Commerce Agent: Constraint-Aware Post-Training with SFT & GRPO

**Trajectory distillation, corrective fine-tuning, and online reinforcement learning for multi-constraint shopping decisions.**

[中文](README.md) · [Results](docs/results.md) · [Design](next/docs/project-completion-plan.md) · [Reproduction](docs/reproduce.md) · [Experiments](docs/experiments.md)

This project trains Qwen3.5-2B to search products, verify requirements, select options, and complete purchases in ShopSimulator. A shared execution protocol connects teacher collection, student rollouts, and evaluation.

## Pipeline

1. Audit 23,421 source tasks and freeze 12,077 independent tasks after quality filtering and product/near-duplicate grouping.
2. Collect 2,000 verified successful teacher trajectories and 200 validation trajectories with Astra.
3. Train action-only SFT, then apply corrective SFT with 144 verified correction trajectories and 144 replay demonstrations. Only the teacher continuation is supervised for corrections.
4. Run terminal-reward LoRA GRPO on two H100 80GB GPUs: two effective task groups per update, four trajectories per task, 500 updates. Use veRL's clipped policy loss, vLLM sampling, consistent reward-based group filtering, and exact per-turn action tokens.
5. Select checkpoints on dev300, compare fixed candidates on test1000, and diagnose stochastic reliability with eight independent executions per dev task.

## Main results

Same test1000 tasks and execution budget; strict success requires a complete, verifiable `gold_purchase`.

| Model | Strict success | Mean Reward v3 | Mean steps |
|---|---:|---:|---:|
| Domain baseline | 1.5% | −0.1039 | 5.748 |
| Cold-start SFT | 75.0% | 0.7034 | 8.800 |
| Corrective SFT | 76.6% | 0.7229 | 10.000 |
| **Terminal GRPO, step500** | **77.8%** | **0.7364** | **8.498** |
| Astra teacher reference | 80.9% | 0.7837 | 10.390 |

The complete pipeline improves on cold-start SFT by **2.8 percentage points**. GRPO adds **1.2 points** over corrective SFT and reduces mean steps by **15.0%**. The success-rate difference is not statistically significant in the paired test (`p=0.2242`). Steps are averaged over all tasks, including failures; they do not establish a matched-success latency improvement. Wrong purchases increase from 12 to 26, so this is not an across-the-board behavioral improvement.

### Stochastic diagnostics — dev300, temperature 1

| Model | pass@1 | pass@8 |
|---|---:|---:|
| Cold-start SFT | 71.79% | 86.67% |
| Corrective SFT | 69.29% | 88.33% |
| **Terminal GRPO** | **74.83%** | **89.33%** |

GRPO improves pass@1 over corrective SFT by **5.54 points**, with a paired task-bootstrap 95% interval of **[3.92, 7.21]**. The pass@8 change is +1.00 point with an interval crossing zero. This supports improved single-sample reliability in this diagnostic, not a demonstrated expansion of capability or multi-seed robustness. pass@8 is oracle-verified coverage, not deployment success.

[All four models and confidence intervals](next/docs/passk/results.md) · [Machine-readable test results](next/results/evaluation/test-comparison/summary.json)

## Repository guide

- `next/src/`: shared execution, collection, and evaluation protocol.
- `next/scripts/`: training, evaluation, diagnostics, and artifact handling.
- `next/configs/`, `next/data/`: experiment recipes and frozen data products.
- `next/results/`, `next/docs/`: machine-readable evidence and phase summaries.
- `environments/`: embedded shopping environment.
- `docs/`: project architecture, main results, experiment and artifact indexes.

The current runnable pipeline remains under `next/` to preserve frozen paths and recovery contracts. Earlier execution code and 200-task experiments are historical assets, not part of the current headline comparison.

## Reproduction and scope

Use the [execution guide](docs/reproduce.md) and exact model/configuration hashes. Checkpoints, raw trajectories, and long logs are externally archived; Git contains code, data products, recipes, and compact evidence. Teacher collection uses a private gateway, and HDFS artifacts require access. Existing teacher products can be reused without fresh collection. This is not an infrastructure-free one-command reproduction.

Local-credit GRPO, larger batches, KL regularization, SDPO, and evidence-constrained rewards remain in the [experiment archive](docs/experiments.md), including negative findings. All reported training results are single-seed; repeated inspection of the fixed test set limits claims of untouched final evaluation. Independent holdouts and multiple training seeds remain future work.

See [third-party sources and licensing status](THIRD_PARTY_NOTICES.md).
