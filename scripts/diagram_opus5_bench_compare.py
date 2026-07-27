"""Grouped bar chart: Frontier-Bench v0.1 and SWE-bench Pro (light theme)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = (Path(__file__).resolve().parents[1]
       / "static/images/opus5-vs-fable5-benchmark-vs-real-tasks/bench-compare.png")

MODELS = ["Claude Opus 5", "Claude Fable 5", "Claude Opus 4.8"]
COLORS = ["#635BFF", "#00B4A6", "#B0B0B8"]

FRONTIER = [43.3, 33.7, 18.7]
SWE_PRO = [79.2, 80.0, 69.2]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="white")
x = np.arange(len(MODELS))

panels = [
    (axes[0], FRONTIER, "Frontier-Bench v0.1", "score", 50),
    (axes[1], SWE_PRO, "SWE-bench Pro", "resolved (%)", 95),
]

for ax, values, title, ylabel, ymax in panels:
    ax.set_facecolor("white")
    bars = ax.bar(x, values, width=0.58, color=COLORS,
                  edgecolor="#E5E5EA", linewidth=0.8, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + ymax * 0.022,
                f"{value}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#0A0A0A", zorder=4)
    ax.set_title(title, fontsize=14, fontweight="bold",
                 color="#0A0A0A", pad=14)
    ax.set_ylabel(ylabel, fontsize=11, color="#6B6B75")
    ax.set_ylim(0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(["Opus 5", "Fable 5", "Opus 4.8"],
                       fontsize=11, color="#0A0A0A")
    ax.tick_params(axis="y", labelsize=10, colors="#6B6B75", length=0)
    ax.grid(axis="y", color="#EAEAEA", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#E5E5EA")

fig.suptitle("Agentic coding benchmarks: Opus 5 vs Fable 5 vs Opus 4.8",
             fontsize=15, fontweight="bold", color="#0A0A0A", y=0.99)
fig.text(0.5, 0.035,
         "Source: Anthropic (Frontier-Bench v0.1, via Vellum)  |  "
         "CodingFleet (SWE-bench Pro)",
         ha="center", fontsize=9.5, color="#6B6B75")
fig.text(0.99, 0.01, "jesamkim.github.io", ha="right", va="bottom",
         fontsize=9, color="#B0B0B8", fontweight="bold")

fig.tight_layout(rect=[0.01, 0.09, 0.99, 0.94])
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, facecolor="white")
print(f"saved: {OUT}")
