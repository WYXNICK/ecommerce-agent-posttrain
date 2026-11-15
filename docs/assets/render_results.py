"""Render publication figures from archived summaries. Requires matplotlib."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
COLORS = {
    "baseline": "#B4BBC3", "s0": "#4878A8", "s-repair": "#D18F3A",
    "grpo-500": "#21856B", "astra": "#95929F", "s0-grpo-450": "#9976A5",
}
LABELS = {
    "baseline": "Domain baseline", "s0": "Cold-start SFT",
    "s-repair": "Corrective SFT", "grpo-500": "Terminal GRPO",
    "astra": "Astra teacher", "s0-grpo-450": "S0 → GRPO",
}


def load(relative):
    return json.loads((ROOT / relative).read_text())


def style(ax, grid_axis="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#D9DDE1", linewidth=0.5, linestyle=(0, (2, 3)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#747474")
    ax.spines["left"].set_color("#747474")
    ax.tick_params(direction="out", length=3, width=0.6, pad=5)


def save(fig, name):
    for suffix in ("svg", "pdf", "png"):
        metadata = {"Date": None} if suffix == "svg" else {}
        if suffix == "pdf":
            metadata = {"CreationDate": None, "ModDate": None, "Title": name.replace("-", " ")}
        target = OUT / f"{name}.{suffix}"
        fig.savefig(target, dpi=300, facecolor="white", metadata=metadata)
        if suffix == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


def test_results(summary):
    scores = summary["summaries"]
    names = ["baseline", "s0", "s-repair", "grpo-500", "astra"]
    positions = [0, 1, 2, 3, 4.3]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25), sharey=True)
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.19, top=0.86, wspace=0.22)

    for ax, metric, scale, limit, ticks, title, label in (
        (axes[0], "strict_success_rate", 100, 100, [0, 20, 40, 60, 80, 100],
         "(a) Strict task success", "Success rate (%)"),
        (axes[1], "average_steps", 1, 12, [0, 3, 6, 9, 12],
         "(b) Interaction length", "Mean steps per task"),
    ):
        for y, name in zip(positions, names):
            value = scores[name][metric] * scale
            reference = name in ("baseline", "astra")
            ax.barh(y, value, height=0.52, color=COLORS[name], alpha=0.65 if reference else 0.90,
                    edgecolor=COLORS[name], linewidth=0.6, hatch="///" if name == "astra" else None)
            number = f"{value:.1f}" if metric == "strict_success_rate" else f"{value:.2f}"
            ax.text(value + limit * 0.018, y, number, va="center", fontsize=10,
                    weight="bold" if name == "grpo-500" else "normal")
        ax.set_xlim(0, limit)
        ax.set_xticks(ticks)
        ax.set_xlabel(label, labelpad=8)
        ax.set_title(title, loc="left", pad=13)
        style(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.axhline(3.7, color="#D1D4D8", linewidth=0.6, linestyle=(0, (3, 4)))
    axes[0].set_yticks(positions, [LABELS[name] for name in names])
    axes[0].set_ylim(4.85, -0.6)
    axes[0].get_yticklabels()[3].set_fontweight("bold")
    save(fig, "test-results")


def sampling_results(summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.29, top=0.86, wspace=0.40)
    ax = axes[0]
    ks = [1, 2, 4, 8]
    for name, marker, line in (
        ("s0", "o", "-"), ("s-repair", "^", "--"),
        ("s0-grpo-450", "D", "-."), ("grpo-500", "s", "-"),
    ):
        values = [100 * summary["models"][name]["pass_at_k"][str(k)]["estimate"] for k in ks]
        ax.plot(ks, values, marker=marker, markersize=4.7,
                markerfacecolor=COLORS[name] if name == "grpo-500" else "white",
                markeredgewidth=1.1, linewidth=1.7 if name == "grpo-500" else 1.3,
                color=COLORS[name], label=LABELS[name], linestyle=line, zorder=3 if name == "grpo-500" else 2)
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.88, 9.1)
    ax.xaxis.set_major_locator(FixedLocator(ks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(65, 95)
    ax.set_yticks([65, 70, 75, 80, 85, 90, 95])
    ax.set_ylabel(r"pass@$k$ (%)", labelpad=8)
    ax.set_xlabel(r"Number of independent attempts, $k$", labelpad=8)
    ax.set_title("(a) Sampling reliability and coverage", loc="left", pad=13)
    style(ax, "y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, -0.28), ncol=2,
              frameon=False, handlelength=2.5, columnspacing=1.4, fontsize=9)

    ax = axes[1]
    comparisons = summary["comparisons"]["s-repair__grpo-500"]["delta_pass_at_k"]
    for y, k in enumerate(("1", "8")):
        item = comparisons[k]
        estimate = item["estimate"] * 100
        low, high = [v * 100 for v in item["ci95_task_bootstrap"]]
        ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]], fmt="o",
                    color=COLORS["grpo-500"], markersize=5.5, capsize=4, linewidth=1.4, capthick=1)
        ax.text(estimate, y + 0.22, f"{estimate:+.2f}  [{low:+.2f}, {high:+.2f}]",
                ha="center", va="top", fontsize=9.5)
    ax.axvline(0, color="#818181", linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_yticks([0, 1], [r"$\Delta$pass@1", r"$\Delta$pass@8"])
    ax.set_ylim(1.55, -0.55)
    ax.set_xlim(-4, 9.5)
    ax.set_xticks([-4, -2, 0, 2, 4, 6, 8])
    ax.set_xlabel("Change (percentage points)", labelpad=8)
    ax.set_title("(b) GRPO − corrective SFT", loc="left", pad=13)
    ax.text(0.5, -0.34, "95% paired task-bootstrap confidence intervals",
            transform=ax.transAxes, ha="center", fontsize=9)
    style(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "sampling-results")


def configure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal",
        "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 10,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#333333", "ytick.color": "#333333", "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path",
        "svg.hashsalt": "ecommerce-agent-results",
    })


if __name__ == "__main__":
    configure_style()
    test_results(load("next/results/evaluation/test-comparison/summary.json"))
    sampling_results(load("next/results/passk/dev300/summary.json"))
