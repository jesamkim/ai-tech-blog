"""Cost comparison chart for the China open-weight rush post.

Vertical stacked layout (portrait ~1600x1800) so labels stay legible at
blog/mobile width.
Top panel: list price per 1M tokens (official pricing pages, 2026-08-08).
Bottom panel: Reuters-reported average cost per task from Artificial Analysis data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# Every fallback must carry Korean glyphs; DejaVu Sans renders tofu boxes.
KOREAN_FONTS = ('NanumSquareRound', 'NanumGothic', 'NanumBarunGothic',
                'NanumSquare', 'Noto Sans CJK KR')
available = {f.name for f in fm.fontManager.ttflist}
family = next((f for f in KOREAN_FONTS if f in available), None)
if family is None:
    raise SystemExit(
        'No Korean-capable font found. Install one of: '
        + ', '.join(KOREAN_FONTS)
        + ' (e.g. apt-get install fonts-nanum fonts-noto-cjk), '
          'then clear the matplotlib font cache.'
    )
plt.rcParams['font.family'] = family
plt.rcParams['axes.unicode_minus'] = False

BG = '#ffffff'
TEXT = '#0A0A0A'
MUTED = '#6B6B75'
FAINT = '#B0B0B8'
GRID = '#EAEAEA'
TEAL = '#00B4A6'
PURPLE = '#635BFF'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), facecolor=BG,
                               gridspec_kw={'height_ratios': [1, 1]})

# ----------------------------------------------------------------- top panel
models = ['DeepSeek\nV4-Flash', 'Claude\nOpus 4.8']
inp = [0.14, 5.00]
out = [0.28, 25.00]
x = np.arange(len(models))
w = 0.34

for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=18, length=0)

b1 = ax1.bar(x - w / 2, inp, w, color=TEAL, label='입력 100만 토큰', zorder=3)
b2 = ax1.bar(x + w / 2, out, w, color=PURPLE, label='출력 100만 토큰', zorder=3)

ax1.set_yscale('log')
ax1.set_ylim(0.08, 90)
ax1.set_yticks([0.1, 1, 10])
ax1.set_yticklabels(['$0.1', '$1', '$10'])
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=19, color=TEXT)
ax1.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
ax1.set_axisbelow(True)
ax1.set_ylabel('100만 토큰당 정가 (로그 스케일)', fontsize=18, color=MUTED, labelpad=10)

for bars, vals in ((b1, inp), (b2, out)):
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.16,
                 f'${val:,.2f}', ha='center', va='bottom',
                 fontsize=21, fontweight='bold', color=TEXT)

ax1.set_title('정가 비교 · 출력 토큰 기준 약 89배 차이', fontsize=25,
              fontweight='bold', color=TEXT, pad=14, loc='left')
leg = ax1.legend(frameon=False, fontsize=18, loc='upper left',
                 bbox_to_anchor=(0.02, 0.93))
for text in leg.get_texts():
    text.set_color(MUTED)

# -------------------------------------------------------------- bottom panel
task_models = ['DeepSeek\nV4-Flash', 'Kimi K3', 'GPT-5.6\nSol', 'Claude\nFable 5']
task_cost = [0.03, 0.86, 1.86, 3.15]
xb = np.arange(len(task_models))

bars = ax2.bar(xb, task_cost, 0.5,
               color=[TEAL, PURPLE, '#B0B0B8', '#8C8C95'], zorder=3)
ax2.set_yscale('log')
ax2.set_ylim(0.018, 12)
ax2.set_yticks([0.03, 0.3, 3])
ax2.set_yticklabels(['$0.03', '$0.30', '$3.00'])
ax2.set_xticks(xb)
ax2.set_xticklabels(task_models, fontsize=19, color=TEXT)
ax2.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
ax2.set_axisbelow(True)
ax2.set_ylabel('테스트 1건당 평균 비용 (로그 스케일)', fontsize=18, color=MUTED, labelpad=10)

for bar, val in zip(bars, task_cost):
    ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.18,
             f'${val:,.2f}', ha='center', va='bottom',
             fontsize=21, fontweight='bold', color=TEXT)

ax2.set_title('테스트 1건당 평균 비용 · 오픈웨이트도 규모에 따라 갈림', fontsize=25, fontweight='bold',
              color=TEXT, pad=14, loc='left')

fig.text(0.5, 0.042,
         '정가 출처: DeepSeek · Anthropic 공식 가격 (2026-08-08 확인, DeepSeek 인상 예고).',
         ha='center', fontsize=15, color=FAINT)
fig.text(0.5, 0.016,
         '태스크 비용 출처: Artificial Analysis 측정치 인용 Reuters 보도 (2026-08-03).',
         ha='center', fontsize=15, color=FAINT)
fig.text(0.988, 0.006, 'jesamkim.github.io', ha='right', fontsize=15,
         color=FAINT, fontweight='bold', alpha=0.55)

fig.tight_layout(rect=(0.01, 0.065, 0.99, 0.98), h_pad=3.4)
out_dir = (Path(__file__).resolve().parent.parent
           / 'static' / 'images' / 'china-open-weight-rush')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'diagram-2-cost-comparison.png'
fig.savefig(out_path, dpi=200, facecolor=BG)
print('saved', out_path)
