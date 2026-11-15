"""Render architecture, data, case, and checkpoint figures from repository evidence."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from render_results import COLORS, configure_style, load, save, style

BLUE, GOLD, GREEN = COLORS["s0"], COLORS["s-repair"], COLORS["grpo-500"]
GRAY, INK = "#77818C", "#222222"


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.98)
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, title, body="", color=BLUE, fill="#FFFFFF", size=10):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.07",
            linewidth=0.8,
            edgecolor=color,
            facecolor=fill,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.23,
        title,
        ha="center",
        va="center",
        weight="bold",
        fontsize=size,
        zorder=3,
    )
    if body:
        ax.text(
            x + w / 2,
            y + (h - 0.40) / 2,
            body,
            ha="center",
            va="center",
            fontsize=size - 0.5,
            linespacing=1.4,
            color="#40464D",
            zorder=3,
        )


def arrow(ax, start, end, color=GRAY, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color=color,
            linestyle="--" if dashed else "-",
            zorder=1,
        )
    )


def architecture():
    fig, ax = canvas(12, 5.8)
    ax.text(0.14, 5.53, "(a) Post-training pipeline", fontsize=12)
    nodes = [
        (
            "Teacher demonstrations",
            "Astra · environment-verified\n2,000 train / 200 validation",
            GRAY,
        ),
        ("Cold-start SFT", "Qwen3.5-2B · action-only CE\nSuccessful demonstrations", BLUE),
        ("Corrective SFT", "144 corrections + 144 replay\nTeacher suffix supervision", GOLD),
        ("Terminal-reward GRPO", "2 tasks × 4 rollouts / update\nLoRA · 500 updates", GREEN),
        ("Evaluation", "dev300 selection · test1000\nIndependent pass@k diagnosis", GRAY),
    ]
    for i, (title, body, color) in enumerate(nodes):
        x = 0.17 + i * 2.4
        box(ax, x, 4.14, 2.05, 1.06, title, body, color, size=10)
        if i < 4:
            arrow(ax, (x + 2.10, 4.67), (x + 2.35, 4.67))
    ax.text(
        4.99,
        3.87,
        "Failure replay → teacher takeover → verified correction",
        ha="center",
        fontsize=9,
        color=GOLD,
    )
    ax.text(0.14, 3.39, "(b) Shared interaction protocol", fontsize=12)
    ax.add_patch(
        FancyBboxPatch(
            (0.16, 1.53),
            11.65,
            1.55,
            boxstyle="round,pad=0.035,rounding_size=0.08",
            facecolor="#F7F8FA",
            edgecolor="#D2D7DC",
            linewidth=0.8,
            zorder=0,
        )
    )
    box(ax, 0.36, 1.97, 1.75, 0.84, "Task + history", "Frozen split / budget", GRAY, size=10)
    box(ax, 2.73, 1.97, 2.05, 0.84, "Policy", "Teacher API / student vLLM", BLUE, size=10)
    box(ax, 5.42, 1.97, 2.08, 0.84, "Action Guard", "Parse · legal target · limits", GOLD, size=10)
    box(
        ax,
        8.15,
        1.97,
        3.28,
        0.84,
        "ShopSimulator",
        "Search → inspect → select → buy",
        GREEN,
        size=10,
    )
    for start, end in [(2.15, 2.67), (4.82, 5.36), (7.54, 8.09)]:
        arrow(ax, (start, 2.39), (end, 2.39))
    ax.plot([9.79, 9.79, 3.76], [1.92, 1.72, 1.72], color=GRAY, lw=0.9, zorder=1)
    arrow(ax, (3.76, 1.72), (3.76, 1.93))
    ax.text(
        6.75,
        1.76,
        "Structured observation + execution feedback",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "#F7F8FA", "edgecolor": "none", "pad": 1},
    )
    ax.text(
        11.70,
        3.37,
        "Collection · correction · rollout · evaluation",
        ha="right",
        fontsize=9,
        color=GRAY,
    )
    ax.text(0.14, 1.15, "(c) Learning signals and evidence", fontsize=12)
    box(
        ax,
        0.17,
        0.10,
        3.57,
        0.78,
        "SFT targets",
        "Verified actions; observations are masked",
        BLUE,
        size=10,
    )
    box(
        ax,
        4.19,
        0.10,
        3.57,
        0.78,
        "GRPO returns",
        "Terminal reward; model failures retained",
        GREEN,
        size=10,
    )
    box(
        ax,
        8.21,
        0.10,
        3.57,
        0.78,
        "Reproducible evidence",
        "Task traces · metrics · checkpoint hashes",
        GRAY,
        size=10,
    )
    save(fig, "system-architecture")


def data_flow(metadata):
    fig, ax = canvas(11.8, 5.5)
    ax.text(0.15, 5.20, "(a) Task qualification and isolation", fontsize=12)
    stages = [
        ("Source tasks", metadata["source_task_count"], "Product / instruction records"),
        ("Quality-qualified", metadata["quality_pass_count"], "Price, options, valid target"),
        ("History-isolated", metadata["fresh_quality_pass_count"], "Exclude prior-exposed groups"),
        (
            "Group representatives",
            metadata["fresh_representative_count"],
            "Product + near-duplicate grouping",
        ),
    ]
    for i, (title, count, note) in enumerate(stages):
        x = 0.18 + i * 2.96
        box(ax, x, 3.96, 2.53, 0.95, title, f"{count:,}\n{note}", GRAY, size=10)
        if i < 3:
            arrow(ax, (x + 2.58, 4.44), (x + 2.89, 4.44))
    ax.text(0.15, 3.51, "(b) Frozen, disjoint partitions", fontsize=12)
    ax.add_patch(
        Rectangle((0.17, 0.85), 5.52, 2.36, facecolor="#F1F6FA", edgecolor="#CBD7E1", lw=0.8)
    )
    ax.add_patch(
        Rectangle((6.11, 0.85), 5.52, 2.36, facecolor="#F5F4F8", edgecolor="#D4D1DC", lw=0.8)
    )
    ax.text(0.37, 2.91, "Training-side tasks", fontsize=11, color=BLUE, weight="bold")
    ax.text(
        6.31, 2.91, "Held out from parameter updates", fontsize=11, color="#766786", weight="bold"
    )
    groups = [
        (0.37, 2.25, "teacher_candidates", "Teacher / correction candidates", BLUE),
        (0.37, 1.70, "grpo_train", "Online GRPO task pool", BLUE),
        (0.37, 1.15, "train_reserve", "Training reserve", BLUE),
        (6.31, 2.25, "sft_validation_candidates", "SFT validation candidates", GRAY),
        (6.31, 1.70, "dev", "Development / checkpoint selection", GRAY),
        (6.31, 1.15, "test", "Fixed test comparison", GRAY),
    ]
    for x, y, key, label, color in groups:
        count = metadata["splits"][key]["tasks"]
        ax.text(x, y, label, fontsize=10.5, va="center")
        ax.text(
            x + 4.90,
            y,
            f"{count:,}",
            fontsize=12,
            va="center",
            ha="right",
            weight="bold",
            color=color,
        )
    ax.plot([5.90, 5.90], [0.78, 3.26], color="#A8A0B5", lw=1, ls=(0, (4, 3)))
    ax.text(
        5.90,
        0.48,
        "Split by task and duplicate group before collection or training.",
        ha="center",
        fontsize=10,
    )
    ax.text(
        5.90,
        0.15,
        "Teacher pilot (100) and dev smoke (50) are subsets, not additional tasks.",
        ha="center",
        fontsize=9,
        color=GRAY,
    )
    save(fig, "data-partitions")


def correction_case(case):
    fig, ax = canvas(11.8, 5.1)
    ax.text(
        0.15,
        4.83,
        f"Task {case['task_id']} · One verified correction from the training data",
        fontsize=12,
    )
    box(
        ax,
        0.20,
        3.79,
        11.38,
        0.70,
        "Request",
        "Khaki summer suit · waistcoat + striped short-sleeve shirt + shorts + plain khaki tie · 90 cm · under CNY 300",
        GRAY,
        size=10,
    )
    xvals = [0.20, 4.16, 8.12]
    headers = ["(a) Student decision", "(b) Teacher intervention", "(c) Environment verification"]
    fills = ["#FBF5ED", "#F0F5FA", "#EDF6F2"]
    colors = [GOLD, BLUE, GREEN]
    for x, title, fill, color in zip(xvals, headers, fills, colors):
        box(ax, x, 1.39, 3.46, 2.04, title, color=color, fill=fill, size=11)
    student, teacher = case["student_terminal"], case["teacher_terminal"]
    prefix_actions = case["shared_prefix_actions"]
    teacher_actions = case["supervised_teacher_actions"]
    budget = teacher["reward_detail"]["hard_gates"]["budget"]["required"]
    ax.text(
        0.43,
        2.84,
        "Selected: khaki waistcoat + shorts + tie\nSize: 90 cm\nMissing: striped short-sleeve shirt",
        va="top",
        fontsize=10,
        linespacing=1.45,
    )
    ax.text(
        0.43,
        1.92,
        f"buy_now() → partial purchase\nCNY {student['purchase']['price']:.0f} · Reward v3 = {student['reward']:.4f}",
        va="top",
        fontsize=10,
        color="#9A6528",
        linespacing=1.5,
    )
    ax.text(
        4.39,
        2.84,
        f"Replay {prefix_actions} actions; intervene before buy.\n1. select_option(full khaki outfit)\n2. buy_now()",
        va="top",
        fontsize=10,
        linespacing=1.45,
    )
    ax.text(
        4.39,
        1.92,
        "Same product; retain the 90 cm size.\nOnly the teacher suffix is supervised.",
        va="top",
        fontsize=10,
        color=BLUE,
        linespacing=1.5,
    )
    ax.text(
        8.35,
        2.84,
        f"All requested garments present\nSize: 90 cm\nCNY {teacher['purchase']['price']:.0f} < CNY {budget:.0f} budget",
        va="top",
        fontsize=10,
        linespacing=1.45,
    )
    ax.text(
        8.35,
        1.92,
        f"gold_purchase · reward_valid = true\nReward v3 = {teacher['reward']:.1f}",
        va="top",
        fontsize=10,
        color=GREEN,
        linespacing=1.5,
    )
    arrow(ax, (3.72, 2.42), (4.09, 2.42), BLUE)
    arrow(ax, (7.68, 2.42), (8.05, 2.42), GREEN)
    ax.text(0.22, 0.93, "Supervision mask", fontsize=10.5, weight="bold")
    for i in range(prefix_actions + teacher_actions):
        x = 2.15 + i * 0.64
        ax.add_patch(
            Rectangle(
                (x, 0.67),
                0.51,
                0.44,
                facecolor="#E4E7EB" if i < prefix_actions else "#D4E8DF",
                edgecolor="#ABB3BB" if i < prefix_actions else GREEN,
                lw=0.7,
            )
        )
        ax.text(x + 0.255, 0.89, str(i + 1), ha="center", va="center", fontsize=9)
    ax.text(4.65, 0.40, "Student prefix: context only", ha="center", fontsize=9, color=GRAY)
    ax.text(8.00, 0.40, "Teacher actions: CE loss", ha="center", fontsize=9, color=GREEN)
    ax.text(
        0.22,
        0.06,
        "English paraphrase of the recorded Chinese task and options; this is teacher repair, not a post-training student result.",
        fontsize=9,
        color=GRAY,
    )
    save(fig, "correction-case")


def checkpoint_curves(summary):
    candidates = sorted(summary["candidates"], key=lambda c: c["step"])
    steps = [c["step"] for c in candidates]
    selected = summary["selected"]["step"]
    selected_index = steps.index(selected)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.15))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.27, top=0.85, wspace=0.39)
    metrics = [
        ("strict_success_rate", 100, "(a) Strict success", "Success rate (%)"),
        ("mean_final_reward", 1, "(b) Terminal reward", "Mean Reward v3"),
        ("average_steps", 1, "(c) Interaction length", "Mean steps per task"),
    ]
    for ax, (key, scale, title, ylabel) in zip(axes, metrics):
        values = [c["summary"][key] * scale for c in candidates]
        ax.plot(steps, values, "o-", color=BLUE, lw=1.3, ms=3.8)
        ax.axhline(values[0], color=GRAY, lw=0.9, ls=(0, (4, 3)))
        ax.scatter([selected], [values[selected_index]], color=GREEN, s=44, zorder=4, marker="s")
        ax.set(
            xlim=(-10, 520),
            xticks=[0, 100, 200, 300, 400, 500],
            xlabel="Policy updates",
            ylabel=ylabel,
        )
        ax.set_title(title, loc="left", pad=12)
        ax.tick_params(axis="x", labelsize=9)
        style(ax, "y")
        ax.margins(y=0.20)
        if key == "mean_final_reward":
            best = max(range(len(values)), key=values.__getitem__)
            ax.annotate(
                f"Peak reward: step {steps[best]}",
                xy=(steps[best], values[best]),
                xytext=(115, values[best] + 0.006),
                fontsize=8.5,
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": GRAY},
            )
    handles = [
        Line2D(
            [], [], marker="o", color=BLUE, lw=1.3, label="dev300 evaluation (every 50 updates)"
        ),
        Line2D([], [], color=GRAY, ls="--", lw=0.9, label="Corrective SFT start"),
        Line2D([], [], marker="s", color=GREEN, ls="", label=f"Selected checkpoint: {selected}"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.015),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    save(fig, "grpo-checkpoints")


if __name__ == "__main__":
    configure_style()
    architecture()
    data_flow(load("next/data/task_pool/metadata.json"))
    correction_case(load("next/results/data-reflow/r1/case-19141.json"))
    checkpoint_curves(load("next/results/grpo/outcome/summary.json"))
