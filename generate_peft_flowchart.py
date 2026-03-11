#!/usr/bin/env python3
"""PEFT Decision Flowchart - 한글 폰트 수정"""

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
BLUE = "#4fc3f7"
GREEN = "#66bb6a"
YELLOW = "#ffd54f"

# Figure 생성 (가로 18, 세로 21, xlim 여유 충분히)
fig, ax = plt.subplots(figsize=(18, 21), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(-1, 17)  # 오른쪽 여유 증가
ax.set_ylim(0, 20)
ax.axis("off")

def draw_diamond(ax, x, y, w, h, text, color=BLUE):
    """다이아몬드 판단 노드"""
    diamond = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=BOX_COLOR, edgecolor=color, linewidth=2
    )
    ax.add_patch(diamond)
    # 텍스트 줄바꿈
    lines = text.split("\n")
    y_offset = 0.15 * (len(lines) - 1)
    for i, line in enumerate(lines):
        ax.text(x, y + y_offset - i * 0.3, line, ha="center", va="center",
                fontsize=10, color=TEXT_COLOR, weight="bold")

def draw_box(ax, x, y, w, h, text, color=GREEN):
    """사각형 결과 노드"""
    box = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=BOX_COLOR, edgecolor=color, linewidth=2
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=11, color=color, weight="bold")

def draw_arrow(ax, x1, y1, x2, y2, label=""):
    """직선 화살표"""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=TEXT_COLOR, lw=2))
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, ha="left", va="center",
                fontsize=9, color=YELLOW, style="italic")

# === 플로우차트 구조 ===

# 시작
ax.text(8, 19, "PEFT 기법 선택 가이드", ha="center", va="center",
        fontsize=16, color=TEXT_COLOR, weight="bold")

# 1. GPU 메모리
draw_diamond(ax, 8, 17, 3, 1, "GPU 메모리\n제한 심각?")
draw_arrow(ax, 8, 16.5, 8, 15.5, "YES")
draw_box(ax, 8, 15, 2.5, 0.8, "QLoRA", GREEN)

draw_arrow(ax, 10, 17, 12, 17)
ax.text(11, 17.3, "NO", ha="center", va="bottom", fontsize=9, color=YELLOW, style="italic")

# 2. 태스크 복잡도
draw_diamond(ax, 12, 17, 3, 1, "태스크\n복잡도 높음?")
draw_arrow(ax, 12, 16.5, 12, 15.5, "YES")
draw_box(ax, 12, 15, 2.5, 0.8, "LoRA / DoRA", GREEN)

draw_arrow(ax, 12, 16.5, 8, 13.5)
ax.text(10, 15, "NO", ha="center", va="center", fontsize=9, color=YELLOW, style="italic")

# 3. 도메인 적응
draw_diamond(ax, 8, 13, 3, 1, "도메인 적응\n목적?")
draw_arrow(ax, 8, 12.5, 8, 11.5, "YES")
draw_box(ax, 8, 11, 2.5, 0.8, "Adapter", GREEN)

draw_arrow(ax, 6, 13, 4, 13)
ax.text(5, 13.3, "NO", ha="center", va="bottom", fontsize=9, color=YELLOW, style="italic")

# 4. 프롬프트 기반
draw_diamond(ax, 4, 13, 3, 1, "프롬프트\n기반 선호?")
draw_arrow(ax, 4, 12.5, 4, 11.5, "YES")

# 5. 생성 태스크
draw_diamond(ax, 4, 11, 2.8, 1, "생성\n태스크?")
draw_arrow(ax, 4, 10.5, 2.5, 9.5, "YES")
draw_box(ax, 2.5, 9, 2.5, 0.8, "Prefix-Tuning", GREEN)

draw_arrow(ax, 4, 10.5, 5.5, 9.5, "NO")
draw_box(ax, 5.5, 9, 2.5, 0.8, "P-Tuning v2", GREEN)

# 6. 추론 속도
draw_arrow(ax, 4, 12.5, 8, 9)
ax.text(6, 10.5, "NO", ha="center", va="center", fontsize=9, color=YELLOW, style="italic")

draw_diamond(ax, 8, 9, 3, 1, "추론 속도\n중요?")
draw_arrow(ax, 8, 8.5, 8, 7.5, "YES")
draw_box(ax, 8, 7, 2.5, 0.8, "IA³", GREEN)

draw_arrow(ax, 10, 9, 12, 9)
ax.text(11, 9.3, "NO", ha="center", va="bottom", fontsize=9, color=YELLOW, style="italic")

# 7. 관리형 서비스
draw_diamond(ax, 12, 9, 3, 1, "관리형\n서비스?")
draw_arrow(ax, 12, 8.5, 12, 7.5, "YES")
draw_box(ax, 12, 7, 2.5, 0.8, "Nova Forge", GREEN)

draw_arrow(ax, 12, 8.5, 12, 5.5, "NO")
draw_box(ax, 12, 5, 2.5, 0.8, "LoRA (기본)", GREEN)

# 하단 요약
ax.text(8, 3, "※ 일반적 권장: LoRA → DoRA → QLoRA 순서로 시도",
        ha="center", va="center", fontsize=11, color=BLUE, style="italic")

# 워터마크
ax.text(0.5, 0.5, "jesamkim.github.io", fontsize=10, color=TEXT_COLOR, alpha=0.4, weight="bold")

plt.tight_layout()
plt.savefig("static/images/peft-decision-flowchart.png", dpi=100, facecolor=BG_COLOR)
print("✓ peft-decision-flowchart.png 생성 완료")
plt.close()
