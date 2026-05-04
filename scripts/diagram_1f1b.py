import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Color palette
BG = '#1e1e2e'
TEXT = '#cdd6f4'
BLUE = '#89b4fa'
GREEN = '#a6e3a1'
RED = '#f38ba8'
YELLOW = '#f9e2af'
PURPLE = '#cba6f7'
TEAL = '#94e2d5'
SUBTEXT = '#a6adc8'
SURFACE = '#313244'
OVERLAY = '#45475a'

# Micro-batch colors
MB_COLORS = {
    0: BLUE,
    1: TEAL,
    2: GREEN,
    3: YELLOW,
}

MB_COLORS_DARK = {
    0: '#5b7ec2',
    1: '#6aab9c',
    2: '#6fb076',
    3: '#c4b06e',
}

def add_watermark(fig):
    fig.text(0.99, 0.01, "jesamkim.github.io",
             ha="right", va="bottom", color="#6c7086",
             fontsize=8, alpha=0.7, fontstyle="italic")

plt.rcParams['font.family'] = 'NanumSquareRound'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

n_gpus = 4

# 1F1B schedule (4 GPUs, 4 micro-batches)
# Warmup: each GPU fills its pipeline depth with forward passes
# Then steady state: 1 forward, 1 backward alternating
# Cooldown: remaining backward passes
#
# GPU 0 (depth=4): F0 F1 F2 F3 B0 B1 B2 B3
# GPU 1 (depth=3): .  F0 F1 F2 B0 F3 B1 B2 B3
# GPU 2 (depth=2): .  .  F0 F1 B0 F2 B1 F3 B2 B3
# GPU 3 (depth=1): .  .  .  F0 B0 F1 B1 F2 B2 F3 B3

# Schedule as (type, micro_batch, time_slot) per GPU
schedules = {
    0: [('F', 0, 0), ('F', 1, 1), ('F', 2, 2), ('F', 3, 3),
        ('B', 0, 4), ('B', 1, 5), ('B', 2, 6), ('B', 3, 7)],
    1: [('F', 0, 1), ('F', 1, 2), ('F', 2, 3),
        ('B', 0, 4), ('F', 3, 5), ('B', 1, 6), ('B', 2, 7), ('B', 3, 8)],
    2: [('F', 0, 2), ('F', 1, 3),
        ('B', 0, 4), ('F', 2, 5), ('B', 1, 6), ('F', 3, 7), ('B', 2, 8), ('B', 3, 9)],
    3: [('F', 0, 3),
        ('B', 0, 4), ('F', 1, 5), ('B', 1, 6), ('F', 2, 7), ('B', 2, 8), ('F', 3, 9), ('B', 3, 10)],
}

total_time = 12
block_w = 0.85
block_h = 0.65
gpu_labels = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']

# Draw blocks
for g in range(n_gpus):
    y_center = n_gpus - 1 - g
    for (typ, m, t) in schedules[g]:
        if typ == 'F':
            color = MB_COLORS[m]
            text_color = BG
        else:
            color = MB_COLORS_DARK[m]
            text_color = TEXT

        rect = FancyBboxPatch(
            (t + 0.075, y_center - block_h/2),
            block_w, block_h,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='white', linewidth=1,
            zorder=3
        )
        ax.add_patch(rect)
        ax.text(t + 0.075 + block_w/2, y_center, f"{typ}{m}",
                ha='center', va='center', color=text_color,
                fontsize=11, fontweight='bold', zorder=4)

# Add "free" markers where activation memory is released (after backward completes)
# After each backward for micro-batch m on GPU g, that micro-batch's activations are freed
free_markers = []
for g in range(n_gpus):
    for (typ, m, t) in schedules[g]:
        if typ == 'B':
            free_markers.append((g, t))

# Only show some representative "free" labels to avoid clutter
# Show on GPU 2 and GPU 3 where the interleaving is most visible
shown_free = set()
for g in range(n_gpus):
    for (typ, m, t) in schedules[g]:
        if typ == 'B' and g >= 1:
            key = (g, t)
            if key not in shown_free and len(shown_free) < 6:
                shown_free.add(key)
                y_center = n_gpus - 1 - g
                ax.text(t + 0.075 + block_w/2, y_center + block_h/2 + 0.12,
                        "free", ha='center', va='bottom',
                        color=GREEN, fontsize=7, fontstyle='italic', zorder=5)

# Draw bubble/idle regions
# For 1F1B, bubbles are much smaller than GPipe
# GPU 0: no idle (continuous from t0-t7)
# GPU 1: idle at t0
# GPU 2: idle at t0, t1
# GPU 3: idle at t0, t1, t2

for g in range(n_gpus):
    y_center = n_gpus - 1 - g
    # Find all occupied time slots
    occupied = set()
    for (typ, m, t) in schedules[g]:
        occupied.add(t)

    # Find all time slots for this GPU's range
    all_times = [entry[2] for entry in schedules[g]]
    min_t_global = 0
    max_t_local = max(all_times)

    for t in range(min_t_global, max_t_local + 1):
        if t not in occupied:
            rect = plt.Rectangle(
                (t + 0.075, y_center - block_h/2),
                block_w, block_h,
                linewidth=1.2, linestyle='--',
                edgecolor=OVERLAY, facecolor='none',
                zorder=1
            )
            ax.add_patch(rect)

# Configure axes
ax.set_xlim(-0.8, total_time - 0.5)
ax.set_ylim(-1.2, n_gpus - 0.1)

ax.set_yticks([n_gpus - 1 - i for i in range(n_gpus)])
ax.set_yticklabels(gpu_labels, fontsize=12, color=TEXT, fontweight='bold')

ax.set_xticks(np.arange(total_time) + 0.5)
ax.set_xticklabels([f't{i}' for i in range(total_time)], fontsize=9, color=SUBTEXT)
ax.set_xlabel("Time Steps", fontsize=12, color=TEXT, labelpad=10)

# Grid lines
for t in range(total_time + 1):
    ax.axvline(x=t, color=SURFACE, linewidth=0.5, zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(OVERLAY)
ax.spines['left'].set_color(OVERLAY)
ax.tick_params(colors=SUBTEXT)

# Title
ax.set_title("1F1B Schedule", fontsize=18, color=TEXT,
             fontweight='bold', pad=20)

# Annotation: Peak Activation Memory reduction
ax.annotate(
    "Peak Activation Memory 감소",
    xy=(5.5, -0.8),
    fontsize=13, color=GREEN, fontweight='bold', ha='center',
    bbox=dict(boxstyle='round,pad=0.4', facecolor=SURFACE, edgecolor=GREEN, alpha=0.9),
    zorder=10
)

# Arrow pointing from annotation to the interleaved region
ax.annotate(
    "",
    xy=(4.5, 0.45), xytext=(5.5, -0.55),
    arrowprops=dict(arrowstyle='->', color=GREEN, lw=2),
    zorder=10
)

# Phase annotations
# Warmup phase bracket
ax.annotate("Warmup", xy=(1.5, n_gpus - 0.25), fontsize=10, color=PURPLE,
            ha='center', va='bottom', fontstyle='italic')

# Steady state
ax.annotate("Steady State (1F1B)", xy=(6, n_gpus - 0.25), fontsize=10, color=TEAL,
            ha='center', va='bottom', fontstyle='italic')

# Cooldown
ax.annotate("Cooldown", xy=(9.5, n_gpus - 0.25), fontsize=10, color=YELLOW,
            ha='center', va='bottom', fontstyle='italic')

# Legend for micro-batches
legend_items = []
for m in range(4):
    fwd = mpatches.Patch(facecolor=MB_COLORS[m], edgecolor='white', label=f'm{m} Forward')
    legend_items.append(fwd)
for m in range(4):
    bwd = mpatches.Patch(facecolor=MB_COLORS_DARK[m], edgecolor='white', label=f'm{m} Backward')
    legend_items.append(bwd)

leg = ax.legend(handles=legend_items, loc='lower right', fontsize=9, ncol=2,
                facecolor=SURFACE, edgecolor=OVERLAY, labelcolor=TEXT)

add_watermark(fig)
plt.tight_layout()

out = '/Workshop/yan/ai-tech-blog/static/images/2026-04-16-pipeline-parallelism-evolution/diagram_1f1b.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

from PIL import Image
img = Image.open(out)
w, h = img.size
assert w >= 1200, f"Too narrow: {w}px"
assert h >= 600, f"Too short: {h}px"
print(f"OK: {out}: {w}x{h}px")
