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


def section(ax, y, letter, title, right=""):
    ax.text(0.25, y, letter, fontsize=10, color=GRAY, va="center")
    ax.text(0.56, y, title, fontsize=12, va="center")
    ax.plot([0.25, 11.55], [y - 0.20, y - 0.20], color="#D7DBDE", lw=0.6)
    if right:
        ax.text(11.55, y, right, fontsize=9, color=GRAY, ha="right", va="center")


def box(ax, x, y, w, h, title, body, color=GRAY):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.04",
            linewidth=0.65,
            edgecolor="#CDD2D6",
            facecolor="white",
            zorder=2,
        )
    )
    ax.plot([x + 0.14, x + 0.52], [y + h, y + h], color=color, lw=2.0, zorder=3)
    ax.text(x + 0.14, y + h - 0.25, title, va="center", fontsize=10.5, weight="bold", zorder=3)
    ax.text(
        x + 0.14,
        y + h - 0.53,
        body,
        va="top",
        fontsize=9.5,
        color="#454B51",
        linespacing=1.45,
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
    fig, ax = canvas(11.8, 5.7)
    section(ax, 5.38, "a", "Post-training pipeline", "Qwen3.5-2B  /  LoRA")
    nodes = [
        ("Demonstrations", "Astra · verified in environment\n2,000 train / 200 validation", GRAY),
        ("Cold-start SFT", "Successful demonstrations\nAction-only cross-entropy", BLUE),
        ("Corrective SFT", "144 corrections + 144 replay\nTeacher suffix supervision", GOLD),
        ("Terminal GRPO", "2 tasks × 4 rollouts / update\n500 policy updates", GREEN),
        ("Evaluation", "dev300 selection · test1000\nIndependent pass@k diagnosis", GRAY),
    ]
    for i, (title, body, color) in enumerate(nodes):
        x = 0.25 + i * 2.32
        box(ax, x, 3.93, 2.02, 1.02, title, body, color)
        if i < 4:
            arrow(ax, (x + 2.06, 4.42), (x + 2.27, 4.42))
    ax.text(
        4.89,
        3.69,
        "Failure replay → teacher takeover → verified correction",
        ha="center",
        fontsize=9,
        color="#9A6E35",
    )

    section(
        ax,
        3.25,
        "b",
        "Shared interaction protocol",
        "Collection · correction · rollout · evaluation",
    )
    box(ax, 0.25, 1.85, 1.85, 1.04, "Task + history", "Frozen task split\nExecution budget", GRAY)
    box(ax, 2.65, 1.85, 2.25, 1.04, "Policy", "Teacher API\nor student vLLM", BLUE)
    box(
        ax,
        5.48,
        1.85,
        2.25,
        1.04,
        "Action Guard",
        "Parse · check legal targets\nApply execution limits",
        GOLD,
    )
    box(
        ax,
        8.31,
        1.85,
        3.24,
        1.04,
        "ShopSimulator",
        "Search → inspect → select → buy\nStructured state and terminal evidence",
        GREEN,
    )
    for start, end in [(2.14, 2.60), (4.94, 5.43), (7.77, 8.26)]:
        arrow(ax, (start, 2.37), (end, 2.37))
    ax.plot([9.93, 9.93, 3.77], [1.85, 1.47, 1.47], color=GRAY, lw=0.9)
    arrow(ax, (3.77, 1.47), (3.77, 1.81))
    ax.text(
        6.87,
        1.48,
        "Observation + execution feedback",
        ha="center",
        va="center",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 4},
    )

    section(ax, 1.01, "c", "Learning signals and evidence")
    columns = [
        (0.25, "SFT targets", "Verified actions; observations masked", BLUE),
        (4.17, "GRPO returns", "Terminal reward; model failures retained", GREEN),
        (8.09, "Evidence", "Task traces · metrics · checkpoint hashes", GRAY),
    ]
    for x, title, body, color in columns:
        ax.plot([x, x], [0.12, 0.62], color=color, lw=1.5)
        ax.text(x + 0.13, 0.53, title, fontsize=10.5, weight="bold")
        ax.text(x + 0.13, 0.20, body, fontsize=9.5, color="#454B51")
    save(fig, "system-architecture")


def data_flow(metadata):
    fig, ax = canvas(11.8, 5.2)
    section(ax, 4.90, "a", "Task qualification", "Counts refer to tasks, not accepted trajectories")
    source = metadata["source_task_count"]
    stages = [
        ("Source tasks", source, "Product / instruction records"),
        ("Quality-qualified", metadata["quality_pass_count"], "Price, options, valid target"),
        ("History-isolated", metadata["fresh_quality_pass_count"], "Exclude prior-exposed groups"),
        (
            "Group representatives",
            metadata["fresh_representative_count"],
            "Product + near-duplicate grouping",
        ),
    ]
    for i, (title, count, note) in enumerate(stages):
        x = 0.25 + i * 2.93
        ax.text(x, 4.37, title, fontsize=10.5, weight="bold")
        ax.text(x, 3.89, f"{count:,}", fontsize=24, color=INK)
        ax.text(x, 3.61, note, fontsize=9, color=GRAY)
        ax.add_patch(Rectangle((x, 3.36), 2.50, 0.065, facecolor="#E6E9EC", edgecolor="none"))
        ax.add_patch(
            Rectangle(
                (x, 3.36),
                2.50 * count / source,
                0.065,
                facecolor=BLUE if i == 3 else "#9BA7B2",
                edgecolor="none",
            )
        )
        if i < 3:
            arrow(ax, (x + 2.53, 4.06), (x + 2.83, 4.06))
    ax.text(
        11.55, 3.12, "Bar lengths relative to the source pool", ha="right", fontsize=8.5, color=GRAY
    )
    section(
        ax, 2.78, "b", "Frozen partitions", "Split by task and duplicate group before collection"
    )
    sides = [
        (
            0.25,
            "Training-side tasks",
            BLUE,
            [
                ("teacher_candidates", "Teacher / correction candidates"),
                ("grpo_train", "Online GRPO task pool"),
                ("train_reserve", "Training reserve"),
            ],
        ),
        (
            6.35,
            "Held out from parameter updates",
            "#766786",
            [
                ("sft_validation_candidates", "SFT validation candidates"),
                ("dev", "Development / checkpoint selection"),
                ("test", "Fixed test comparison"),
            ],
        ),
    ]
    for x, title, color, rows in sides:
        ax.text(x, 2.20, title, fontsize=11, color=color, weight="bold")
        ax.text(x + 5.20, 2.20, "TASKS", ha="right", fontsize=8, color=GRAY)
        for i, (key, label) in enumerate(rows):
            y = 1.78 - i * 0.51
            ax.text(x, y, label, fontsize=10.5, va="center")
            ax.text(
                x + 5.20,
                y,
                f"{metadata['splits'][key]['tasks']:,}",
                ha="right",
                va="center",
                fontsize=13,
                color=color,
            )
            ax.plot([x, x + 5.20], [y - 0.22, y - 0.22], color="#E0E3E6", lw=0.55)
    ax.plot([5.95, 5.95], [0.50, 2.23], color="#BAC2C9", lw=0.7, ls=(0, (3, 4)))
    ax.text(
        0.25,
        0.16,
        "Teacher pilot (100) ⊂ teacher candidates; dev smoke (50) ⊂ dev. Subsets are not counted twice.",
        fontsize=9,
        color=GRAY,
    )
    save(fig, "data-partitions")


def correction_case(case):
    fig, ax = canvas(11.8, 5.4)
    student, teacher = case["student_terminal"], case["teacher_terminal"]
    prefix_actions = case["shared_prefix_actions"]
    teacher_actions = case["supervised_teacher_actions"]
    budget = teacher["reward_detail"]["hard_gates"]["budget"]["required"]
    section(ax, 5.09, "a", "A verified correction", f"Training task {case['task_id']}")
    ax.text(0.25, 4.59, "REQUEST", fontsize=8.5, color=GRAY)
    ax.text(
        1.25,
        4.60,
        "Khaki summer suit: waistcoat, striped short-sleeve shirt, shorts and plain khaki tie.",
        fontsize=11,
    )
    ax.text(1.25, 4.30, "Size 90 cm  ·  Budget under CNY 300", fontsize=10, color="#454B51")

    columns = [
        (0.25, "Student decision", "#A5793D"),
        (4.19, "Teacher intervention", BLUE),
        (8.13, "Verified outcome", GREEN),
    ]
    for i, (x, title, color) in enumerate(columns):
        ax.plot([x, x + 3.42], [3.94, 3.94], color=color, lw=1.2)
        ax.text(x, 3.66, title, fontsize=12, weight="bold")
        if i < 2:
            arrow(ax, (x + 3.49, 3.71), (x + 3.85, 3.71))
    ax.text(
        0.25,
        3.24,
        "Waistcoat + shorts + tie\nSize: 90 cm",
        fontsize=10.5,
        va="top",
        linespacing=1.5,
    )
    ax.text(
        0.25,
        2.56,
        "Missing: striped short-sleeve shirt",
        fontsize=10.5,
        color="#93622E",
        bbox={"facecolor": "#FAF2E6", "edgecolor": "none", "pad": 4},
    )
    ax.text(
        0.25, 2.16, f"CNY {student['purchase']['price']:.0f}  ·  Partial purchase", fontsize=10.5
    )
    ax.text(0.25, 1.81, f"Reward v3  {student['reward']:.4f}", fontsize=11, color="#93622E")

    ax.text(
        4.19,
        3.24,
        f"Replay {prefix_actions} actions; intervene before buy.\nSame product and 90 cm size.",
        fontsize=10.5,
        va="top",
        linespacing=1.5,
    )
    for y, number, text in [
        (2.54, "1", "Select the full outfit option"),
        (2.16, "2", "Execute buy_now()"),
    ]:
        ax.text(4.19, y, number, color=BLUE, fontsize=10.5, weight="bold")
        ax.text(4.46, y, text, fontsize=10.5)
    ax.text(4.19, 1.81, "Only the teacher suffix is supervised", fontsize=10, color=BLUE)

    ax.text(
        8.13,
        3.24,
        "All requested garments present\nSize: 90 cm",
        fontsize=10.5,
        va="top",
        linespacing=1.5,
    )
    ax.text(
        8.13,
        2.56,
        f"CNY {teacher['purchase']['price']:.0f} < CNY {budget:.0f} budget",
        fontsize=10.5,
    )
    ax.text(8.13, 2.16, "gold_purchase · reward_valid = true", fontsize=10, color=GREEN)
    ax.text(8.13, 1.81, f"Reward v3  {teacher['reward']:.1f}", fontsize=11, color=GREEN)

    ax.plot([0.25, 11.55], [1.53, 1.53], color="#D7DBDE", lw=0.6)
    ax.text(0.25, 1.17, "b", fontsize=10, color=GRAY)
    ax.text(0.56, 1.17, "Supervision mask", fontsize=11)
    start, pitch, width, y = 3.05, 0.69, 0.57, 1.04
    for i in range(prefix_actions + teacher_actions):
        x = start + i * pitch
        is_teacher = i >= prefix_actions
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                0.31,
                facecolor="#D4E8DF" if is_teacher else "#EDF0F2",
                edgecolor=GREEN if is_teacher else "#C5CCD1",
                lw=0.65,
            )
        )
        ax.text(
            x + width / 2,
            y + 0.155,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=8.5,
            color=GREEN if is_teacher else "#606972",
        )
    for first, count, label, color in [
        (0, prefix_actions, "Student prefix · context only", GRAY),
        (prefix_actions, teacher_actions, "Teacher actions · CE loss", GREEN),
    ]:
        left, right = start + first * pitch, start + (first + count - 1) * pitch + width
        ax.plot([left, left, right, right], [0.97, 0.88, 0.88, 0.97], color=color, lw=0.6)
        ax.text((left + right) / 2, 0.62, label, ha="center", fontsize=9, color=color)
    ax.text(
        0.25,
        0.17,
        "English paraphrase of the recorded Chinese task. Teacher repair example; not a post-training student comparison.",
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
