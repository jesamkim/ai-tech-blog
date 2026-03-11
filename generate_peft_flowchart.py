#!/usr/bin/env python3
"""PEFT Decision Flowchart - 완전 새로 작성"""
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
YELLOW = "#ffd54f"
PURPLE = "#ab47bc"

fig, ax = plt.subplots(figsize=(16, 20), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-1, 15)
ax.set_ylim(-1, 19)
ax.axis("off")

def draw_box(x, y, w, h, text, color=TX, sub=None):
    box = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                   facecolor=BX, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2+(0.15 if sub else 0), text, ha="center", va="center",
            fontsize=11, color=color, weight="bold")
    if sub:
        ax.text(x+w/2, y+h/2-0.25, sub, ha="center", va="center",
                fontsize=9, color=TX, alpha=0.8)

def draw_diamond(x, y, w, h, text, sub=None):
    cx, cy = x+w/2, y+h/2
    pts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    diamond = plt.Polygon(pts, facecolor=BX, edgecolor=YELLOW, linewidth=2)
    ax.add_patch(diamond)
    ax.text(cx, cy+(0.15 if sub else 0), text, ha="center", va="center",
            fontsize=10, color=YELLOW, weight="bold")
    if sub:
        ax.text(cx, cy-0.25, sub, ha="center", va="center", fontsize=8, color=TX)

def arrow(x1, y1, x2, y2, color=TX):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5))

# Title
ax.text(7, 18.3, "PEFT 기법 선택 가이드", ha="center", fontsize=16, color=TX, weight="bold")

# Start
draw_box(4.5, 17, 5, 0.8, "파인튜닝 필요", GREEN)

# Q1: GPU 메모리
arrow(7, 17, 7, 16.2)
draw_diamond(4.5, 14.8, 5, 1.4, "GPU 메모리 제약?", "(VRAM < 16GB)")

# Q1 YES -> QLoRA
ax.text(3.8, 15.2, "YES", fontsize=9, color=GREEN, weight="bold")
arrow(4.5, 15.5, 2.5, 15.5)
draw_box(0.2, 14.8, 2.3, 1, "QLoRA", PURPLE, "4-bit 양자화+LoRA")

# Q1 NO -> Q2
ax.text(10, 15.2, "NO", fontsize=9, color=RED, weight="bold")
arrow(9.5, 15.5, 11, 15.5)

# Q2: Forgetting 우려?
draw_diamond(8.5, 13, 5, 1.4, "Forgetting 우려?", "(범용 능력 유지 필요)")

arrow(11, 14.8, 11, 14.4)

# Q2 YES -> LoRA + Data Mixing
ax.text(7.8, 13.4, "YES", fontsize=9, color=GREEN, weight="bold")
arrow(8.5, 13.7, 6.5, 13.7)
draw_box(3, 13, 3.5, 1, "LoRA/DoRA", GREEN, "+ Data Mixing")

# Q2 NO -> Q3
ax.text(11.5, 12.5, "NO", fontsize=9, color=RED, weight="bold")
arrow(11, 13, 11, 11.5)

# Q3: 데이터 크기
draw_diamond(8.5, 10, 5, 1.4, "데이터 크기?")

# <5K
ax.text(7.8, 10.4, "<5K", fontsize=9, color=BLUE, weight="bold")
arrow(8.5, 10.7, 6.5, 10.7)
draw_box(3.5, 10, 3, 1, "LoRA", BLUE, "r=16, alpha=64")

# 5K-50K
ax.text(10.5, 9.5, "5K-50K", fontsize=9, color=BLUE, weight="bold")
arrow(11, 10, 11, 8.5)
draw_box(9, 7.8, 4, 0.8, "LoRA+ (r=32)", BLUE)

# >50K
ax.text(14, 10.4, ">50K", fontsize=9, color=RED, weight="bold")
arrow(13.5, 10.7, 14, 10.7)
draw_box(12.5, 9.5, 2.3, 0.8, "Full-rank", RED, "또는 DoRA")

# Bottom recommendation
ax.text(7, 5.5, "일반적 권장 순서", ha="center", fontsize=12, color=YELLOW, weight="bold")
ax.text(7, 4.8, "LoRA (r=16) → DoRA → LoRA+ (r=32) → QLoRA (메모리 부족 시)", 
        ha="center", fontsize=10, color=TX)

# Nova Forge box
nf = mpatches.FancyBboxPatch((2, 2.5, ), 10, 1.5, boxstyle="round,pad=0.2",
                              facecolor=BX, edgecolor=BLUE, linewidth=2)
ax.add_patch(nf)
ax.text(7, 3.7, "Amazon Nova Forge 추천 설정", ha="center", fontsize=11, color=BLUE, weight="bold")
ax.text(7, 3.1, "LoRA+ (alpha=64, lora_plus_lr_ratio=64) + Data Mixing 활성화", 
        ha="center", fontsize=9, color=TX)

# Watermark
ax.text(0.5, -0.5, "jesamkim.github.io", fontsize=10, color=TX, alpha=0.4)

plt.tight_layout()
plt.savefig("static/images/peft-decision-flowchart.png", dpi=100, facecolor=BG, bbox_inches="tight")
print("Done: peft-decision-flowchart.png")
plt.close()
