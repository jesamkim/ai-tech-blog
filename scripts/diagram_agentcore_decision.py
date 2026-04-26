"""Decision tree: Managed Harness vs Code-defined vs Bedrock Agents."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = '#1a1a2e'
PANEL = '#16213e'
TEXT = '#e0e0e0'
SUBTEXT = '#9aa5ce'
BLUE = '#4fc3f7'
GREEN = '#66bb6a'
RED = '#ef5350'
PURPLE = '#ab47bc'
YELLOW = '#ffd54f'
ORANGE = '#ff9800'
BORDER = '#30365f'

plt.rcParams['font.family'] = 'NanumSquareRound'
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 8.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(7, 8.45, '어떤 방식을 선택할 것인가',
        ha='center', va='center', color=TEXT, fontsize=17, fontweight='bold')
ax.text(7, 8.02, 'Managed Harness vs Code-defined Agent vs Bedrock Agents',
        ha='center', va='center', color=SUBTEXT, fontsize=10, style='italic')

# Start node
start_box = FancyBboxPatch((5.5, 7.1), 3, 0.6,
                           boxstyle='round,pad=0.02,rounding_size=0.1',
                           linewidth=1.5, edgecolor=BLUE, facecolor=PANEL)
ax.add_patch(start_box)
ax.text(7, 7.4, '에이전트가 필요하다',
        ha='center', va='center', color=BLUE, fontsize=11, fontweight='bold')

# Q1
q1_box = FancyBboxPatch((4.5, 5.9), 5, 0.7,
                        boxstyle='round,pad=0.02,rounding_size=0.1',
                        linewidth=1.5, edgecolor=YELLOW, facecolor=PANEL)
ax.add_patch(q1_box)
ax.text(7, 6.37, 'Q1. agent loop를 직접 구현해야 하는 제약이 있는가?',
        ha='center', va='center', color=YELLOW, fontsize=10, fontweight='bold')
ax.text(7, 6.08, '(온프레미스, 특정 프레임워크, 규제 등)',
        ha='center', va='center', color=SUBTEXT, fontsize=8.5)

# Arrow from start to Q1
ax.annotate('', xy=(7, 6.6), xytext=(7, 7.1),
            arrowprops=dict(arrowstyle='->', color=BORDER, lw=1.3))

# Branch left: YES -> Code-defined (arrow tip lands on Code-defined box top)
ax.annotate('', xy=(2.4, 3.3), xytext=(5.0, 5.9),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.3,
                            connectionstyle='arc3,rad=-0.1'))
ax.text(2.9, 5.0, 'YES', ha='center', va='center', color=RED, fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, edgecolor='none'))

# Right branch goes to Q2
ax.annotate('', xy=(7, 5.1), xytext=(7, 5.9),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.3))
ax.text(7.25, 5.5, 'NO', ha='left', va='center', color=GREEN, fontsize=9, fontweight='bold')

# Q2
q2_box = FancyBboxPatch((4.5, 4.4), 5, 0.7,
                        boxstyle='round,pad=0.02,rounding_size=0.1',
                        linewidth=1.5, edgecolor=YELLOW, facecolor=PANEL)
ax.add_patch(q2_box)
ax.text(7, 4.87, 'Q2. 선언적 설정만으로 충분한가?',
        ha='center', va='center', color=YELLOW, fontsize=10, fontweight='bold')
ax.text(7, 4.58, '(model + systemPrompt + tools 조합으로 커버 가능)',
        ha='center', va='center', color=SUBTEXT, fontsize=8.5)

# Q2 NO -> Bedrock Agents (right, arrow tip lands on BA box top)
ax.annotate('', xy=(11.6, 3.3), xytext=(9.5, 4.4),
            arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.3,
                            connectionstyle='arc3,rad=0.1'))
ax.text(11.35, 4.15, 'NO, 복잡한\nworkflow 필요',
        ha='center', va='center', color=PURPLE, fontsize=8.5, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.25', facecolor=BG, edgecolor='none'))

# Q2 YES label (the arrow is drawn below, spanning the full path)
ax.text(7.3, 3.85, 'YES', ha='left', va='center', color=GREEN, fontsize=9, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor=BG, edgecolor='none'))

# Three answer boxes
box_h = 2.3
box_w = 3.8

# Code-defined (left)
code_box = FancyBboxPatch((0.5, 1.0), box_w, box_h,
                          boxstyle='round,pad=0.02,rounding_size=0.1',
                          linewidth=1.8, edgecolor=RED, facecolor='#2a0e13')
ax.add_patch(code_box)
ax.text(0.5 + box_w/2, 3.05, 'Code-defined Agent',
        ha='center', va='center', color=RED, fontsize=12, fontweight='bold')
ax.text(0.5 + box_w/2, 2.72, 'Strands SDK 등으로 직접 구현',
        ha='center', va='center', color=SUBTEXT, fontsize=8.5, style='italic')

code_pros = [
    'Python 코드 풀 컨트롤',
    '복잡한 분기·루프 자유롭게',
    '온프레미스·하이브리드 가능',
    '디버깅 도구 선택 자유',
]
for i, t in enumerate(code_pros):
    ax.text(0.7, 2.4 - i * 0.3, f'• {t}',
            ha='left', va='center', color=TEXT, fontsize=8.5)

# Managed Harness (center)
mh_box = FancyBboxPatch((5.1, 1.0), box_w, box_h,
                        boxstyle='round,pad=0.02,rounding_size=0.1',
                        linewidth=2, edgecolor=GREEN, facecolor='#0d2818')
ax.add_patch(mh_box)
ax.text(5.1 + box_w/2, 3.05, 'Managed Harness',
        ha='center', va='center', color=GREEN, fontsize=12, fontweight='bold')
ax.text(5.1 + box_w/2, 2.72, '3 선언 + CLI 배포 — Preview',
        ha='center', va='center', color=SUBTEXT, fontsize=8.5, style='italic')

mh_pros = [
    'microVM 세션·영속 FS 무료 제공',
    'CDK 배포 자동 생성',
    '모델·도구·프롬프트 설정 변경',
    'Strands Agents 기반 동작',
]
for i, t in enumerate(mh_pros):
    ax.text(5.3, 2.4 - i * 0.3, f'• {t}',
            ha='left', va='center', color=TEXT, fontsize=8.5)

# Bedrock Agents (right)
ba_box = FancyBboxPatch((9.7, 1.0), box_w, box_h,
                        boxstyle='round,pad=0.02,rounding_size=0.1',
                        linewidth=1.8, edgecolor=PURPLE, facecolor='#1c0e28')
ax.add_patch(ba_box)
ax.text(9.7 + box_w/2, 3.05, 'Bedrock Agents',
        ha='center', va='center', color=PURPLE, fontsize=12, fontweight='bold')
ax.text(9.7 + box_w/2, 2.72, 'action groups + KB 기반',
        ha='center', va='center', color=SUBTEXT, fontsize=8.5, style='italic')

ba_pros = [
    'OpenAPI action groups 중심',
    'Knowledge Bases 내장 연동',
    'Guardrails·감사로그 관리형',
    '엔터프라이즈 워크플로우',
]
for i, t in enumerate(ba_pros):
    ax.text(9.9, 2.4 - i * 0.3, f'• {t}',
            ha='left', va='center', color=TEXT, fontsize=8.5)

# Arrow Q2 YES -> Managed Harness (top-center of MH box)
ax.annotate('', xy=(5.1 + box_w/2, 3.3), xytext=(7, 4.4),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.3))

# Bottom legend bar
bar_y = 0.45
ax.text(7, bar_y, '선택 기준: 제어 수준 · 구현 난이도 · 운영 책임 범위',
        ha='center', va='center', color=SUBTEXT, fontsize=9, style='italic')

# Watermark
fig.text(0.99, 0.012, 'jesamkim.github.io',
         ha='right', va='bottom', color='#6c7086',
         fontsize=8, alpha=0.7, fontstyle='italic')

plt.tight_layout()
out_path = '/Workshop/yan/ai-tech-blog/static/images/bedrock-agentcore-managed-harness-deep-dive/decision-tree.png'
plt.savefig(out_path, dpi=150, facecolor=BG, bbox_inches='tight')
print(f'Saved: {out_path}')
