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

# Darker versions for backward
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
total_time = 16

# GPipe schedule:
# Forward: each GPU processes m0,m1,m2,m3 sequentially; staggered by 1
# Then all forwards done, backward in reverse micro-batch order
#
# GPU 0: F0@0 F1@1 F2@2 F3@3  ---- B3@8  B2@9  B1@10 B0@11
# GPU 1: ---  F0@1 F1@2 F2@3 F3@4  --- B3@9  B2@10 B1@11 B0@12
# GPU 2: ---  ---  F0@2 F1@3 F2@4 F3@5  B3@10 B2@11 B1@12 B0@13
# GPU 3: ---  ---  ---  F0@3 F1@4 F2@5 F3@6 B3@7 B2@8 B1@9 B0@10

# Let me redo: GPipe with 4 stages, 4 micro-batches
# Forward phase: micro-batch m goes through GPU g at time g+m
# GPU 0: F0@0, F1@1, F2@2, F3@3
# GPU 1: F0@1, F1@2, F2@3, F3@4
# GPU 2: F0@2, F1@3, F2@4, F3@5
# GPU 3: F0@3, F1@4, F2@5, F3@6
#
# Backward phase starts after all forwards complete (t=7)
# Backward: micro-batch processes in reverse order through GPUs in reverse
# GPU 3: B3@7, B2@8, B1@9, B0@10
# GPU 2: B3@8, B2@9, B1@10, B0@11
# GPU 1: B3@9, B2@10, B1@11, B0@12
# GPU 0: B3@10, B2@11, B1@12, B0@13

forward_schedule = {}
for g in range(4):
    for m in range(4):
        t = g + m
        forward_schedule[(g, m)] = t

backward_schedule = {}
# Backward: GPU 3 starts at t=7 with B3, B2, B1, B0
# Each GPU starts 1 later
for g in range(4):
    for m_idx, m in enumerate([3, 2, 1, 0]):
        t = 7 + (3 - g) + m_idx
        backward_schedule[(g, m)] = t

block_w = 0.85
block_h = 0.65
gpu_labels = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']

# Find bubble regions per GPU
for g in range(n_gpus):
    y_center = n_gpus - 1 - g
    # Find last forward time and first backward time for this GPU
    f_times = [forward_schedule[(g, m)] for m in range(4)]
    b_times = [backward_schedule[(g, m)] for m in range(4)]
    last_f = max(f_times) + 1  # end of last forward
    first_b = min(b_times)     # start of first backward

    if first_b > last_f:
        bubble_x = last_f + 0.075
        bubble_w = first_b - last_f - 0.15
        rect = plt.Rectangle(
            (bubble_x, y_center - block_h/2),
            bubble_w, block_h,
            linewidth=1.5, linestyle='--',
            edgecolor=OVERLAY, facecolor='none',
            zorder=1
        )
        ax.add_patch(rect)
        mid_x = bubble_x + bubble_w / 2
        ax.text(mid_x, y_center, "Bubble",
                ha='center', va='center', color=OVERLAY,
                fontsize=9, fontstyle='italic', zorder=2)

# Draw forward blocks
for (g, m), t in forward_schedule.items():
    y_center = n_gpus - 1 - g
    color = MB_COLORS[m]
    rect = FancyBboxPatch(
        (t + 0.075, y_center - block_h/2),
        block_w, block_h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor='white', linewidth=1,
        zorder=3
    )
    ax.add_patch(rect)
    ax.text(t + 0.075 + block_w/2, y_center, f"F{m}",
            ha='center', va='center', color=BG,
            fontsize=11, fontweight='bold', zorder=4)

# Draw backward blocks
for (g, m), t in backward_schedule.items():
    y_center = n_gpus - 1 - g
    color = MB_COLORS_DARK[m]
    rect = FancyBboxPatch(
        (t + 0.075, y_center - block_h/2),
        block_w, block_h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor='white', linewidth=1,
        zorder=3
    )
    ax.add_patch(rect)
    ax.text(t + 0.075 + block_w/2, y_center, f"B{m}",
            ha='center', va='center', color=TEXT,
            fontsize=11, fontweight='bold', zorder=4)

# Configure axes
ax.set_xlim(-0.8, total_time - 1.5)
ax.set_ylim(-1.0, n_gpus - 0.1)

ax.set_yticks([n_gpus - 1 - i for i in range(n_gpus)])
ax.set_yticklabels(gpu_labels, fontsize=12, color=TEXT, fontweight='bold')

max_t = 14
ax.set_xticks(np.arange(max_t) + 0.5)
ax.set_xticklabels([f't{i}' for i in range(max_t)], fontsize=9, color=SUBTEXT)
ax.set_xlabel("Time Steps", fontsize=12, color=TEXT, labelpad=10)

# Grid lines
for t in range(max_t + 1):
    ax.axvline(x=t, color=SURFACE, linewidth=0.5, zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(OVERLAY)
ax.spines['left'].set_color(OVERLAY)
ax.tick_params(colors=SUBTEXT)

# Title
ax.set_title("GPipe: Micro-batch Pipeline", fontsize=18, color=TEXT,
             fontweight='bold', pad=20)

# Separator line between forward and backward phases
ax.axvline(x=7, color=PURPLE, linewidth=1.5, linestyle=':', zorder=1, alpha=0.7)
ax.text(7, n_gpus - 0.3, "Forward | Backward", ha='center', va='bottom',
        color=PURPLE, fontsize=10, fontstyle='italic')

# Legend for micro-batches
legend_items = []
for m in range(4):
    fwd = mpatches.Patch(facecolor=MB_COLORS[m], edgecolor='white', label=f'm{m} Forward')
    legend_items.append(fwd)
for m in range(4):
    bwd = mpatches.Patch(facecolor=MB_COLORS_DARK[m], edgecolor='white', label=f'm{m} Backward')
    legend_items.append(bwd)
bubble_patch = mpatches.Patch(facecolor='none', edgecolor=OVERLAY, linestyle='--', label='Bubble (Idle)')
legend_items.append(bubble_patch)

leg = ax.legend(handles=legend_items, loc='lower right', fontsize=9, ncol=3,
                facecolor=SURFACE, edgecolor=OVERLAY, labelcolor=TEXT)

add_watermark(fig)
plt.tight_layout()

out = '/Workshop/yan/ai-tech-blog/static/images/2026-04-16-pipeline-parallelism-evolution/diagram_gpipe.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

from PIL import Image
img = Image.open(out)
w, h = img.size
assert w >= 1200, f"Too narrow: {w}px"
assert h >= 600, f"Too short: {h}px"
print(f"OK: {out}: {w}x{h}px")
