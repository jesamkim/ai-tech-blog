#!/usr/bin/env python3
"""LoRA Architecture - Yan 직접 작성"""
import matplotlib
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

BG = "#1a1a2e"
TX = "#e0e0e0"
BX = "#16213e"
BLUE = "#4fc3f7"
GREEN = "#66bb6a"
RED = "#ef5350"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor=BG)

for ax in [ax1, ax2]:
    ax.set_facecolor(BG)
    ax.axis("off")

# === Helper ===
def box(ax, x, y, w, h, label, color=TX, sublabel=None, dashed=False):
    style = "round,pad=0.1"
    ls = "--" if dashed else "-"
    b = mpatches.FancyBboxPatch((x, y), w, h, boxstyle=style,
                                 facecolor=BX, edgecolor=color, linewidth=2.5, linestyle=ls)
    ax.add_patch(b)
    ty = y + h/2 + (0.15 if sublabel else 0)
    ax.text(x+w/2, ty, label, ha="center", va="center", fontsize=13, color=color, weight="bold")
    if sublabel:
        ax.text(x+w/2, y+h/2-0.2, sublabel, ha="center", va="center", fontsize=9, color=TX, alpha=0.7)

def arr(ax, x1, y1, x2, y2, color=TX):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))

# === Left: Full Fine-Tuning ===
ax1.set_xlim(-0.5, 10)
ax1.set_ylim(0, 8)
ax1.text(5, 7.3, "Full Fine-Tuning", ha="center", fontsize=15, color=RED, weight="bold")

box(ax1, 1, 3.5, 2, 1, "Input", TX)
box(ax1, 4.5, 3.5, 2, 1, "W", RED, "(전체 학습)")
box(ax1, 7.5, 3.5, 2, 1, "Output", TX)

arr(ax1, 3, 4, 4.5, 4)
arr(ax1, 6.5, 4, 7.5, 4)

# Description
ax1.text(5.5, 2, "모든 파라미터 업데이트", ha="center", fontsize=10, color=TX, alpha=0.7)
ax1.text(5.5, 1.5, "Catastrophic Forgetting 위험", ha="center", fontsize=10, color=RED, alpha=0.8)

# === Right: LoRA ===
ax2.set_xlim(-0.5, 12)
ax2.set_ylim(0, 8)
ax2.text(6, 7.3, "LoRA (Low-Rank Adaptation)", ha="center", fontsize=15, color=GREEN, weight="bold")

# Input
box(ax2, 0.5, 3.5, 2, 1, "Input", TX)

# W0 frozen (top path)
box(ax2, 4, 5.2, 2.2, 1, "W0", TX, "(고정)", dashed=True)

# B x A trainable (bottom path)
box(ax2, 4, 1.8, 1, 1, "B", GREEN, "(d x r)")
box(ax2, 5.5, 1.8, 1, 1, "A", GREEN, "(r x k)")

# + circle
circle = plt.Circle((8, 4), 0.35, facecolor=BX, edgecolor=BLUE, linewidth=2)
ax2.add_patch(circle)
ax2.text(8, 4, "+", ha="center", va="center", fontsize=16, color=BLUE, weight="bold")

# Output
box(ax2, 9.5, 3.5, 2, 1, "Output", TX)

# Arrows - top path
arr(ax2, 2.5, 4.2, 4, 5.7, TX)
arr(ax2, 6.2, 5.7, 7.65, 4.2, TX)

# Arrows - bottom path
arr(ax2, 2.5, 3.8, 4, 2.3, GREEN)
arr(ax2, 5, 2.3, 5.5, 2.3, GREEN)
arr(ax2, 6.5, 2.3, 7.65, 3.8, GREEN)

# Arrow to output
arr(ax2, 8.35, 4, 9.5, 4, BLUE)

# Description
ax2.text(6, 0.7, "0.1~1% 파라미터만 학습", ha="center", fontsize=10, color=GREEN, alpha=0.8)
ax2.text(6, 0.2, "원본 가중치 보존 = Forgetting 방지", ha="center", fontsize=10, color=TX, alpha=0.7)

# Equation box
eq_box = mpatches.FancyBboxPatch((2, -0.8), 8, 0.7, boxstyle="round,pad=0.1",
                                  facecolor=BX, edgecolor=BLUE, linewidth=1.5)
# skip equation for cleanliness

# Main title
fig.text(0.5, 0.95, "Full Fine-Tuning vs LoRA 아키텍처 비교", ha="center",
         fontsize=17, color=TX, weight="bold")

# Watermark
fig.text(0.02, 0.02, "jesamkim.github.io", fontsize=9, color=TX, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.93], pad=1)
plt.savefig("static/images/lora-architecture.png", dpi=120, facecolor=BG, bbox_inches="tight")
print("Done: lora-architecture")
plt.close()
