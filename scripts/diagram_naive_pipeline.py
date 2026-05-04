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

def add_watermark(fig):
    fig.text(0.99, 0.01, "jesamkim.github.io",
             ha="right", va="bottom", color="#6c7086",
             fontsize=8, alpha=0.7, fontstyle="italic")

plt.rcParams['font.family'] = 'NanumSquareRound'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

n_gpus = 4
total_time = 9  # t0 to t8

# Timeline:
# GPU 0: F at t=0, idle t1-t6, B at t=7
# GPU 1: idle t0, F at t=1, idle t2-t5, B at t=6, idle t7
# GPU 2: idle t0-t1, F at t=2, idle t3-t4, B at t=5, idle t6-t7
# GPU 3: idle t0-t2, F at t=3, B at t=4, idle t5-t7

forward_times = [0, 1, 2, 3]
backward_times = [7, 6, 5, 4]

block_w = 0.85
block_h = 0.6

gpu_labels = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3']

# Draw idle/bubble regions first
for i in range(n_gpus):
    f_end = forward_times[i] + 1
    b_start = backward_times[i]
    if b_start > f_end:
        bubble_x = f_end + 0.075
        bubble_w = (b_start - f_end) * block_w + (b_start - f_end - 1) * (1 - block_w) + 0.85 - 0.15
        # Calculate properly
        bubble_x = f_end + 0.075
        bubble_w = b_start - f_end - 0.15
        y_center = n_gpus - 1 - i
        rect = plt.Rectangle(
            (bubble_x, y_center - block_h/2),
            bubble_w, block_h,
            linewidth=1.5, linestyle='--',
            edgecolor=OVERLAY, facecolor='none',
            zorder=1
        )
        ax.add_patch(rect)
        # Add "Bubble" label
        mid_x = bubble_x + bubble_w / 2
        ax.text(mid_x, y_center, "Bubble",
                ha='center', va='center', color=OVERLAY,
                fontsize=10, fontstyle='italic', zorder=2)

# Draw forward and backward blocks
for i in range(n_gpus):
    y_center = n_gpus - 1 - i

    # Forward block
    ft = forward_times[i]
    rect_f = FancyBboxPatch(
        (ft + 0.075, y_center - block_h/2),
        block_w, block_h,
        boxstyle="round,pad=0.05",
        facecolor=BLUE, edgecolor='white', linewidth=1.2,
        zorder=3
    )
    ax.add_patch(rect_f)
    ax.text(ft + 0.075 + block_w/2, y_center, "F",
            ha='center', va='center', color=BG,
            fontsize=14, fontweight='bold', zorder=4)

    # Backward block
    bt = backward_times[i]
    rect_b = FancyBboxPatch(
        (bt + 0.075, y_center - block_h/2),
        block_w, block_h,
        boxstyle="round,pad=0.05",
        facecolor=RED, edgecolor='white', linewidth=1.2,
        zorder=3
    )
    ax.add_patch(rect_b)
    ax.text(bt + 0.075 + block_w/2, y_center, "B",
            ha='center', va='center', color=BG,
            fontsize=14, fontweight='bold', zorder=4)

# Configure axes
ax.set_xlim(-0.8, total_time + 1.2)
ax.set_ylim(-0.8, n_gpus - 0.3)

ax.set_yticks([n_gpus - 1 - i for i in range(n_gpus)])
ax.set_yticklabels(gpu_labels, fontsize=12, color=TEXT, fontweight='bold')

ax.set_xticks(np.arange(total_time) + 0.5)
ax.set_xticklabels([f't{i}' for i in range(total_time)], fontsize=10, color=SUBTEXT)
ax.set_xlabel("Time Steps", fontsize=12, color=TEXT, labelpad=10)

# Grid lines
for t in range(total_time + 1):
    ax.axvline(x=t, color=SURFACE, linewidth=0.5, zorder=0)
for i in range(n_gpus):
    y = n_gpus - 1 - i
    ax.axhline(y=y - block_h/2 - 0.1, color=SURFACE, linewidth=0.5, zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color(OVERLAY)
ax.spines['left'].set_color(OVERLAY)
ax.tick_params(colors=SUBTEXT)

# Title
ax.set_title("Naive Pipeline Parallelism", fontsize=18, color=TEXT,
             fontweight='bold', pad=20)

# Legend
legend_f = mpatches.Patch(facecolor=BLUE, edgecolor='white', label='Forward (F)')
legend_b = mpatches.Patch(facecolor=RED, edgecolor='white', label='Backward (B)')
legend_bubble = mpatches.Patch(facecolor='none', edgecolor=OVERLAY, linestyle='--', label='Bubble (Idle)')
ax.legend(handles=[legend_f, legend_b, legend_bubble],
          loc='upper right', fontsize=10,
          facecolor=SURFACE, edgecolor=OVERLAY, labelcolor=TEXT)

# GPU Utilization annotation
ax.annotate("GPU Utilization: ~25%",
            xy=(4.5, -0.6), fontsize=13, color=YELLOW,
            fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=SURFACE, edgecolor=YELLOW, alpha=0.9))

# Arrows showing flow direction
# Forward flow: down
ax.annotate("", xy=(0.5, 0.45), xytext=(0.5, 3.45),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=2, linestyle='--'),
            zorder=0)
ax.text(-0.55, 1.5, "Forward\n  Flow", fontsize=9, color=BLUE,
        ha='center', va='center', rotation=90)

# Backward flow: up
ax.annotate("", xy=(8.5, 2.55), xytext=(8.5, -0.45),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2, linestyle='--'),
            zorder=0)
ax.text(9.1, 1.5, "Backward\n   Flow", fontsize=9, color=RED,
        ha='center', va='center', rotation=90)

add_watermark(fig)
plt.tight_layout()

out = '/Workshop/yan/ai-tech-blog/static/images/2026-04-16-pipeline-parallelism-evolution/diagram_naive_pipeline.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

from PIL import Image
img = Image.open(out)
w, h = img.size
assert w >= 1200, f"Too narrow: {w}px"
assert h >= 600, f"Too short: {h}px"
print(f"OK: {out}: {w}x{h}px")
