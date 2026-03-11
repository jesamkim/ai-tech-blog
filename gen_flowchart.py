#!/usr/bin/env python3
"""PEFT Decision Flowchart - Yan 직접 작성 v2"""
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

fig, ax = plt.subplots(figsize=(14, 18), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-1, 13)
ax.set_ylim(-2, 18)
ax.axis("off")

def box(x, y, w, h, text, color=TX, sub=None):
    b = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                 facecolor=BX, edgecolor=color, linewidth=2)
    ax.add_patch(b)
    ty = y + h/2 + (0.15 if sub else 0)
    ax.text(x+w/2, ty, text, ha="center", va="center", fontsize=11, color=color, weight="bold")
    if sub:
        ax.text(x+w/2, y+h/2-0.2, sub, ha="center", va="center", fontsize=8, color=TX, alpha=0.8)

def diamond(x, y, w, h, text, sub=None):
    cx, cy = x+w/2, y+h/2
    pts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    d = plt.Polygon(pts, facecolor=BX, edgecolor=YELLOW, linewidth=2)
    ax.add_patch(d)
    ax.text(cx, cy+(0.12 if sub else 0), text, ha="center", va="center",
            fontsize=10, color=YELLOW, weight="bold")
    if sub:
        ax.text(cx, cy-0.22, sub, ha="center", va="center", fontsize=8, color=TX, alpha=0.7)

def arr(x1, y1, x2, y2, color=TX):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))

def label(x, y, text, color=GREEN):
    ax.text(x, y, text, fontsize=9, color=color, weight="bold")

# Title
ax.text(6, 17.2, "PEFT 기법 선택 가이드", ha="center", fontsize=16, color=TX, weight="bold")

# Start
box(3.5, 15.8, 5, 0.8, "파인튜닝 필요", GREEN)

# Arrow down
arr(6, 15.8, 6, 15.2)

# Q1: Memory
diamond(3.5, 13.5, 5, 1.6, "GPU 메모리 제약?", "(VRAM < 24GB)")

# Q1-YES -> QLoRA
label(2.5, 14.1, "YES", GREEN)
arr(3.5, 14.3, 1.8, 14.3)
box(0, 13.7, 1.8, 1, "QLoRA", PURPLE, "4-bit + LoRA")

# Q1-NO -> Q2
label(9, 14.1, "NO", RED)
arr(8.5, 14.3, 9.5, 14.3)

# small connector down
arr(9.5, 14.3, 9.5, 12.8)

# Q2: Forgetting
diamond(7, 11.2, 5, 1.6, "Forgetting 우려?", "(범용 능력 유지)")

# Q2-YES -> LoRA + Data Mixing
label(6, 11.0, "YES", GREEN)
arr(7, 12, 5, 12)
box(1.5, 11.3, 3.5, 1, "LoRA/DoRA", GREEN, "+ Data Mixing")

# Q2-NO -> Q3
label(10.2, 10.8, "NO", RED)
arr(9.5, 11.2, 9.5, 10)

# small connector
arr(9.5, 10, 6, 10)

# Q3: Data size
diamond(3.5, 8.3, 5, 1.6, "데이터 크기?")

# <5K
label(2.3, 9, "<5K", BLUE)
arr(3.5, 9.1, 1.8, 9.1)
box(0, 8.5, 1.8, 0.9, "LoRA", BLUE, "r=16")

# 5K-50K
label(5.5, 7.9, "5K~50K", BLUE)
arr(6, 8.3, 6, 7.2)
box(4, 6.5, 4, 0.8, "LoRA+", BLUE, "r=32, alpha=128")

# >50K
label(9, 9, ">50K", RED)
arr(8.5, 9.1, 10, 9.1)
box(10, 8.5, 2.5, 0.9, "Full-rank", RED, "또는 DoRA")

# Divider
ax.plot([0, 12], [5.5, 5.5], color="#333355", linewidth=1, alpha=0.5)

# Bottom: recommendation
ax.text(6, 4.8, "일반적 권장 순서", ha="center", fontsize=13, color=YELLOW, weight="bold")
ax.text(6, 4.1, "LoRA (r=16)  →  DoRA  →  LoRA+ (r=32)  →  QLoRA (메모리 부족 시)",
        ha="center", fontsize=10, color=TX)

# Nova Forge box
nf = mpatches.FancyBboxPatch((1.5, 1.5), 9, 2, boxstyle="round,pad=0.2",
                              facecolor=BX, edgecolor=BLUE, linewidth=2)
ax.add_patch(nf)
ax.text(6, 3, "Amazon Nova Forge 추천 설정", ha="center", fontsize=12, color=BLUE, weight="bold")
ax.text(6, 2.3, "PEFT: LoRA+ (alpha=64, lora_plus_lr_ratio=64)",
        ha="center", fontsize=9, color=TX)
ax.text(6, 1.9, "Data Mixing 활성화  |  Warmup 15%  |  LR 1e-5",
        ha="center", fontsize=9, color=TX, alpha=0.8)

# Watermark
ax.text(0, -1.5, "jesamkim.github.io", fontsize=9, color=TX, alpha=0.3)

plt.tight_layout(pad=0.5)
plt.savefig("static/images/peft-decision-flowchart.png", dpi=120, facecolor=BG, bbox_inches="tight")
print("Done: flowchart")
plt.close()
