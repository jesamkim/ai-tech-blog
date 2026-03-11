#!/usr/bin/env python3
"""PEFT Evolution Timeline 생성 - 한글 폰트 수정"""

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
RED = "#ef5350"
PURPLE = "#ab47bc"
YELLOW = "#ffd54f"

# 타임라인 데이터 (년도, 기법명, 한글설명, 색상, 위치)
timeline_data = [
    (2019, "Adapter", "사전학습 모델에 작은 어댑터 추가", BLUE, 1),
    (2019, "LoRA", "저랭크 행렬 분해로 파라미터 압축", GREEN, -1),
    (2020, "Prefix-Tuning", "입력 프롬프트에 학습 가능한 prefix 추가", RED, 1),
    (2021, "P-Tuning v2", "모든 레이어에 학습 가능한 프롬프트", PURPLE, -1),
    (2022, "IA³", "학습된 벡터를 activation에 곱셈 적용", YELLOW, 1),
    (2023, "QLoRA", "4비트 양자화 + LoRA 결합", BLUE, -1),
    (2024, "DoRA", "Weight 분해를 방향/크기로 개선", GREEN, 1),
    (2025, "Nova Forge", "AWS 관리형 파인튜닝 서비스", RED, -1),
]

# Figure 생성 (가로 21, 세로 12)
fig, ax = plt.subplots(figsize=(21, 12), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
ax.set_xlim(2018.5, 2025.5)
ax.set_ylim(-3, 3)
ax.axis("off")

# 중앙 타임라인 (수평선)
ax.plot([2019, 2025], [0, 0], color=TEXT_COLOR, linewidth=3)

# 각 연도 마커
for year in range(2019, 2026):
    ax.plot([year, year], [-0.1, 0.1], color=TEXT_COLOR, linewidth=2)
    ax.text(year, -0.4, str(year), ha="center", va="top",
            fontsize=14, color=TEXT_COLOR, weight="bold")

# 기법 박스 그리기
for year, name, desc, color, pos in timeline_data:
    y_box = 1.2 if pos == 1 else -1.2
    y_line_end = 0.2 if pos == 1 else -0.2

    # 박스
    box = mpatches.FancyBboxPatch(
        (year - 0.4, y_box - 0.3), 0.8, 0.6,
        boxstyle="round,pad=0.05",
        facecolor=BOX_COLOR, edgecolor=color, linewidth=2
    )
    ax.add_patch(box)

    # 기법명 (영문)
    ax.text(year, y_box + 0.08, name, ha="center", va="center",
            fontsize=13, color=color, weight="bold")

    # 한글 설명
    ax.text(year, y_box - 0.12, desc, ha="center", va="center",
            fontsize=10, color=TEXT_COLOR)

    # 점선 (박스 외부에서 타임라인까지)
    ax.plot([year, year], [y_line_end, y_box - 0.3 if pos == -1 else y_box + 0.3],
            color=TEXT_COLOR, linestyle="--", linewidth=1, alpha=0.5)

# 타이틀
ax.text(2022, 2.5, "PEFT 기법 진화 타임라인 (2019~2025)",
        ha="center", va="center", fontsize=18, color=TEXT_COLOR, weight="bold")

# 워터마크
ax.text(2019.2, -2.7, "jesamkim.github.io",
        fontsize=10, color=TEXT_COLOR, alpha=0.4, weight="bold")

plt.tight_layout()
plt.savefig("static/images/peft-evolution-timeline.png", dpi=100, facecolor=BG_COLOR)
print("✓ peft-evolution-timeline.png 생성 완료")
plt.close()
