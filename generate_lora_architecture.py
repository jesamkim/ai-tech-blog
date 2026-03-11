#!/usr/bin/env python3
"""LoRA Architecture 비교 다이어그램 - 한글 폰트 수정"""

import matplotlib
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 폰트 매니저 캐시 초기화
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

# 다크 테마 색상
BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
BOX_COLOR = "#16213e"
RED = "#ef5350"
GREEN = "#66bb6a"

# Figure 생성 (가로 21, 세로 9)
fig, ax = plt.subplots(figsize=(21, 9), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(0, 20)
ax.set_ylim(0, 8)
ax.axis("off")

# === 좌측: Full Fine-Tuning ===
# Input 박스
input1 = mpatches.FancyBboxPatch((1, 3.5), 1.5, 1, boxstyle="round,pad=0.1",
                                  facecolor=BOX_COLOR, edgecolor=TEXT_COLOR, linewidth=2)
ax.add_patch(input1)
ax.text(1.75, 4, "Input", ha="center", va="center", fontsize=12, color=TEXT_COLOR, weight="bold")

# W 박스 (빨간색)
w1 = mpatches.FancyBboxPatch((4, 3.5), 1.5, 1, boxstyle="round,pad=0.1",
                              facecolor=BOX_COLOR, edgecolor=RED, linewidth=3)
ax.add_patch(w1)
ax.text(4.75, 4, "W", ha="center", va="center", fontsize=14, color=RED, weight="bold")
ax.text(4.75, 3.2, "(전체 학습)", ha="center", va="top", fontsize=9, color=TEXT_COLOR)

# Output 박스
output1 = mpatches.FancyBboxPatch((7, 3.5), 1.5, 1, boxstyle="round,pad=0.1",
                                   facecolor=BOX_COLOR, edgecolor=TEXT_COLOR, linewidth=2)
ax.add_patch(output1)
ax.text(7.75, 4, "Output", ha="center", va="center", fontsize=12, color=TEXT_COLOR, weight="bold")

# 화살표 (직선)
ax.annotate("", xy=(4, 4), xytext=(2.5, 4),
            arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))
ax.annotate("", xy=(7, 4), xytext=(5.5, 4),
            arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))

# 타이틀
ax.text(4.75, 6, "Full Fine-Tuning", ha="center", va="center",
        fontsize=14, color=TEXT_COLOR, weight="bold")

# === 우측: LoRA ===
# Input 박스
input2 = mpatches.FancyBboxPatch((11, 3.5), 1.5, 1, boxstyle="round,pad=0.1",
                                  facecolor=BOX_COLOR, edgecolor=TEXT_COLOR, linewidth=2)
ax.add_patch(input2)
ax.text(11.75, 4, "Input", ha="center", va="center", fontsize=12, color=TEXT_COLOR, weight="bold")

# W0 박스 (frozen, dashed)
w0 = mpatches.FancyBboxPatch((13.5, 4.5), 1.2, 0.8, boxstyle="round,pad=0.1",
                              facecolor=BG_COLOR, edgecolor=TEXT_COLOR, linewidth=2, linestyle="--")
ax.add_patch(w0)
ax.text(14.1, 4.9, "W0", ha="center", va="center", fontsize=12, color=TEXT_COLOR, weight="bold")
ax.text(14.1, 4.3, "(frozen)", ha="center", va="center", fontsize=8, color=TEXT_COLOR, style="italic")

# B*A 박스 (초록색)
ba = mpatches.FancyBboxPatch((13.5, 2.7), 1.2, 0.8, boxstyle="round,pad=0.1",
                              facecolor=BOX_COLOR, edgecolor=GREEN, linewidth=3)
ax.add_patch(ba)
ax.text(14.1, 3.1, "B × A", ha="center", va="center", fontsize=12, color=GREEN, weight="bold")
ax.text(14.1, 2.5, "(학습)", ha="center", va="top", fontsize=8, color=TEXT_COLOR)

# + 기호
ax.text(15.5, 4, "+", ha="center", va="center", fontsize=20, color=TEXT_COLOR, weight="bold")

# Output 박스
output2 = mpatches.FancyBboxPatch((16.5, 3.5), 1.5, 1, boxstyle="round,pad=0.1",
                                   facecolor=BOX_COLOR, edgecolor=TEXT_COLOR, linewidth=2)
ax.add_patch(output2)
ax.text(17.25, 4, "Output", ha="center", va="center", fontsize=12, color=TEXT_COLOR, weight="bold")

# 화살표 (직선)
ax.annotate("", xy=(13.5, 4.9), xytext=(12.5, 4),
            arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))
ax.annotate("", xy=(13.5, 3.1), xytext=(12.5, 4),
            arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))
ax.annotate("", xy=(16.5, 4), xytext=(14.7, 4.9),
            arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))
ax.annotate("", xy=(16.5, 4), xytext=(14.7, 3.1),
            arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))

# 타이틀
ax.text(14.5, 6, "LoRA (Low-Rank Adaptation)", ha="center", va="center",
        fontsize=14, color=TEXT_COLOR, weight="bold")

# 전체 타이틀
ax.text(10, 7.5, "Full Fine-Tuning vs LoRA 아키텍처 비교", ha="center", va="center",
        fontsize=16, color=TEXT_COLOR, weight="bold")

# 워터마크
ax.text(0.5, 0.3, "jesamkim.github.io", fontsize=10, color=TEXT_COLOR, alpha=0.4, weight="bold")

plt.tight_layout()
plt.savefig("static/images/lora-architecture.png", dpi=100, facecolor=BG_COLOR)
print("✓ lora-architecture.png 생성 완료")
plt.close()
