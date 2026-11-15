"""Render README figures from archived summaries. Requires matplotlib."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COLORS = {"baseline": "#A6B0BC", "s0": "#4977AB", "s-repair": "#CA8C36",
          "grpo-500": "#168275", "astra": "#7A718F", "s0-grpo-450": "#9B8BAB"}
LABELS = {"baseline": "Domain baseline", "s0": "Cold-start SFT",
          "s-repair": "Corrective SFT", "grpo-500": "Terminal GRPO",
          "astra": "Astra teacher", "s0-grpo-450": "S0 → GRPO"}


def load(relative):
    return json.loads((ROOT / relative).read_text())


def style(ax, grid_axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#E8EDF2", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=8, colors="#526175")


def save(fig, name):
    for suffix in ("svg", "png"):
        metadata = {"Date": None} if suffix == "svg" else {}
        target = OUT / f"{name}.{suffix}"
        fig.savefig(target, dpi=180, facecolor="white", metadata=metadata)
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


def test_results(summary):
    scores = summary["summaries"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.145, right=0.95, bottom=0.20, top=0.72, wspace=0.48)
    fig.text(0.04, 0.92, "Shopping task performance", fontsize=21, weight="bold", color="#172B42")
    fig.text(0.04, 0.85, f"{summary['tasks']:,} fixed test tasks · Shared execution budget · Reward v3",
             fontsize=11, color="#526175")

    ax = axes[0]
    names = ["baseline", "s0", "s-repair", "grpo-500", "astra"]
    values = [scores[name]["strict_success_rate"] * 100 for name in names]
    ax.barh(range(len(names)), values, height=0.55, color=[COLORS[n] for n in names])
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Strict task success ↑", loc="left", pad=18, weight="bold")
    for y, value in enumerate(values):
        ax.text(value + 1.5, y, f"{value:.1f}%", va="center", weight="bold", color="#172B42")
    style(ax)

    ax = axes[1]
    names = ["s0", "s-repair", "grpo-500"]
    values = [scores[name]["average_steps"] for name in names]
    ax.barh(range(len(names)), values, height=0.45, color=[COLORS[n] for n in names])
    ax.set_yticks(range(len(names)), [LABELS[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 12)
    ax.set_xticks([0, 4, 8, 12])
    ax.set_title("Mean interaction steps ↓", loc="left", pad=18, weight="bold")
    for y, value in enumerate(values):
        ax.text(value + 0.2, y, f"{value:.2f}", va="center", weight="bold", color="#172B42")
    style(ax)
    delta = 100 * (scores["grpo-500"]["strict_success_rate"] - scores["s-repair"]["strict_success_rate"])
    reduction = 100 * (1 - scores["grpo-500"]["average_steps"] / scores["s-repair"]["average_steps"])
    fig.text(0.04, 0.085, f"GRPO vs. corrective SFT: +{delta:.1f} percentage points success · {reduction:.1f}% fewer mean steps",
             fontsize=11, color=COLORS["grpo-500"], weight="bold")
    fig.text(0.04, 0.035, "Success gain is not statistically significant. Steps include failed tasks; they are not latency measurements.",
             fontsize=9, color="#526175")
    save(fig, "test-results")


def sampling_results(summary):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(left=0.075, right=0.95, bottom=0.27, top=0.72, wspace=0.34)
    fig.text(0.04, 0.92, "Single-sample reliability and success coverage", fontsize=21, weight="bold", color="#172B42")
    fig.text(0.04, 0.85, f"dev{summary['tasks']} · {summary['samples_per_task']} independent executions per task · Temperature 1.0 · top-p 1.0",
             fontsize=11, color="#526175")
    ax = axes[0]
    ks = [1, 2, 4, 8]
    for name in ("s0", "s-repair", "s0-grpo-450", "grpo-500"):
        values = [100 * summary["models"][name]["pass_at_k"][str(k)]["estimate"] for k in ks]
        ax.plot(range(4), values, marker="o", markersize=5, linewidth=2.3 if name == "grpo-500" else 1.7,
                color=COLORS[name], label=LABELS[name], linestyle="--" if name == "s0-grpo-450" else "-")
    ax.set_xticks(range(4), [f"pass@{k}" for k in ks])
    ax.set_ylim(65, 95)
    ax.set_yticks([65, 75, 85, 95])
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title("Success within k independent attempts", loc="left", pad=18, weight="bold")
    style(ax, "y")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.19), ncol=2, frameon=False, fontsize=9.5)

    ax = axes[1]
    comparisons = summary["comparisons"]["s-repair__grpo-500"]["delta_pass_at_k"]
    for y, k in enumerate(("1", "8")):
        item = comparisons[k]
        estimate = item["estimate"] * 100
        low, high = [v * 100 for v in item["ci95_task_bootstrap"]]
        ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]], fmt="o",
                    color=COLORS["grpo-500"], markersize=8, capsize=5, linewidth=2.3)
        ax.text(estimate, y + 0.20, f"{estimate:+.2f}  [{low:+.2f}, {high:+.2f}]", ha="center", fontsize=10, color="#172B42")
    ax.axvline(0, color="#A6B0BC", linewidth=1, linestyle="--")
    ax.set_yticks([0, 1], ["Δpass@1", "Δpass@8"])
    ax.set_ylim(1.55, -0.45)
    ax.set_xlim(-4, 10)
    ax.set_xticks([-4, 0, 4, 8])
    ax.set_xlabel("Percentage points · 95% paired bootstrap CI", labelpad=12)
    ax.set_title("GRPO gain over corrective SFT", loc="left", pad=18, weight="bold")
    style(ax)
    fig.text(0.04, 0.035, "pass@8 measures oracle-verified coverage, not deployment success. Intervals resample tasks, not training seeds.",
             fontsize=9, color="#526175")
    save(fig, "sampling-results")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.labelcolor": "#526175", "text.color": "#172B42",
                         "svg.hashsalt": "ecommerce-agent-results"})
    test_results(load("next/results/evaluation/test-comparison/summary.json"))
    sampling_results(load("next/results/passk/dev300/summary.json"))
