#!/usr/bin/env python3
"""PEFT Evolution Timeline - Yan 직접 작성"""
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
CYAN = "#26c6da"

fig, ax = plt.subplots(figsize=(20, 10), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(-0.5, 19.5)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(10, 9.3, "PEFT 기법 진화 타임라인", ha="center", fontsize=18, color=TX, weight="bold")

# Timeline bar
ax.plot([1, 18.5], [5, 5], color="#555555", linewidth=3, solid_capstyle="round")

# Year markers
years = [(1, "2019"), (4.5, "2021"), (8, "2022"), (11.5, "2023"), (15, "2024"), (18, "2025")]
for x, label in years:
    ax.plot(x, 5, "o", color="#888888", markersize=8, zorder=5)
    ax.text(x, 4.3, label, ha="center", fontsize=11, color="#aaaaaa", weight="bold")

# Items: (x, name, color, description, above=True)
items = [
    (1, "Adapter", CYAN, "사전학습 모델에\n작은 어댑터 추가", True),
    (4.5, "LoRA", GREEN, "저랭크 행렬 분해\n메모리 효율적", False),
    (8, "Prompt\nTuning", YELLOW, "입력 프롬프트만\n학습", True),
    (11.5, "QLoRA", PURPLE, "4비트 양자화\n+ LoRA 결합", False),
    (14, "DoRA", RED, "방향/크기 분리\n정확도 향상", True),
    (16, "GaLore", BLUE, "그래디언트 투영\n메모리 절감", True),
    (18, "Nova\nForge", GREEN, "AWS 관리형\n파인튜닝", False),
]

for x, name, color, desc, above in items:
    if above:
        # Box above timeline
        bw, bh = 2.2, 1.4
        by = 6.2
        # Vertical connector
        ax.plot([x, x], [5.3, by], "--", color=color, linewidth=1, alpha=0.5)
        ax.plot(x, 5.3, "o", color=color, markersize=6, zorder=5)
    else:
        bw, bh = 2.2, 1.4
        by = 2.4
        ax.plot([x, x], [4.7, by + bh], "--", color=color, linewidth=1, alpha=0.5)
        ax.plot(x, 4.7, "o", color=color, markersize=6, zorder=5)
    
    bx = x - bw/2
    box = mpatches.FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0.15",
                                   facecolor=BX, edgecolor=color, linewidth=2)
    ax.add_patch(box)
    ax.text(x, by + bh - 0.35, name, ha="center", va="center",
            fontsize=11, color=color, weight="bold")
    ax.text(x, by + 0.35, desc, ha="center", va="center",
            fontsize=8, color=TX, alpha=0.85, linespacing=1.3)

# Watermark
ax.text(0.5, 0.3, "jesamkim.github.io", fontsize=9, color=TX, alpha=0.3)

plt.tight_layout(pad=0.5)
plt.savefig("static/images/peft-evolution-timeline.png", dpi=120, facecolor=BG, bbox_inches="tight")
print("Done: timeline")
plt.close()
