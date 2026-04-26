"""Stack comparison: Traditional agent dev vs Managed Harness."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

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

fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

ax.text(7, 7.5, '에이전트 개발 스택 비교',
        ha='center', va='center', color=TEXT, fontsize=18, fontweight='bold')
ax.text(7, 7.05, 'Traditional Stack vs Managed Harness',
        ha='center', va='center', color=SUBTEXT, fontsize=11, style='italic')

# LEFT: Traditional
left_x = 0.3
right_x = 7.5
col_w = 6.2

ax.text(left_x + col_w/2, 6.4, '기존 방식 (Traditional)',
        ha='center', va='center', color=RED, fontsize=14, fontweight='bold')
ax.text(left_x + col_w/2, 6.05, '직접 구성하고 배포 스크립트까지 작성',
        ha='center', va='center', color=SUBTEXT, fontsize=9)

traditional_layers = [
    ('애플리케이션 코드', 'Agent loop, 프롬프트 튜닝', BLUE),
    ('오케스트레이션 프레임워크', 'LangChain / LlamaIndex / Custom', PURPLE),
    ('세션·메모리 계층', 'Redis, DynamoDB, 상태 직렬화', YELLOW),
    ('도구 통합', 'Code execution, Browser, API wrappers', ORANGE),
    ('인증·식별자', 'IAM, Cognito, API keys, SigV4', GREEN),
    ('런타임·배포', 'ECS/EKS/Lambda + IaC + 모니터링', RED),
]

y_top = 5.6
h = 0.7
gap = 0.1
for i, (title, detail, color) in enumerate(traditional_layers):
    y = y_top - i * (h + gap)
    box = FancyBboxPatch((left_x + 0.1, y - h), col_w - 0.2, h,
                         boxstyle='round,pad=0.02,rounding_size=0.08',
                         linewidth=1.2, edgecolor=color, facecolor=PANEL)
    ax.add_patch(box)
    ax.text(left_x + 0.35, y - h/2 + 0.1, title,
            ha='left', va='center', color=TEXT, fontsize=10.5, fontweight='bold')
    ax.text(left_x + 0.35, y - h/2 - 0.15, detail,
            ha='left', va='center', color=SUBTEXT, fontsize=8.5)
    ax.text(left_x + col_w - 0.3, y - h/2, f'계층 {i+1}',
            ha='right', va='center', color=color, fontsize=8, style='italic')

# Divider
ax.plot([7.2, 7.2], [0.5, 6.4], color=BORDER, linewidth=1, linestyle='--', alpha=0.6)

# RIGHT: Managed Harness
ax.text(right_x + col_w/2 - 0.3, 6.4, 'Managed Harness',
        ha='center', va='center', color=GREEN, fontsize=14, fontweight='bold')
ax.text(right_x + col_w/2 - 0.3, 6.05, '3개의 선언만 있으면 배포 완료',
        ha='center', va='center', color=SUBTEXT, fontsize=9)

# Declaration box (top)
decl_y = 5.3
decl_h = 1.5
decl = FancyBboxPatch((right_x + 0.1, decl_y - decl_h), col_w - 0.4, decl_h,
                     boxstyle='round,pad=0.02,rounding_size=0.08',
                     linewidth=1.5, edgecolor=GREEN, facecolor='#0d2818')
ax.add_patch(decl)
ax.text(right_x + 0.35, decl_y - 0.25, '개발자가 선언하는 것',
        ha='left', va='center', color=GREEN, fontsize=10, fontweight='bold')
ax.text(right_x + 0.35, decl_y - 0.6, 'model',
        ha='left', va='center', color=BLUE, fontsize=10.5, family='monospace', fontweight='bold')
ax.text(right_x + 1.5, decl_y - 0.6, '어떤 모델을 쓸지',
        ha='left', va='center', color=TEXT, fontsize=9.5)
ax.text(right_x + 0.35, decl_y - 0.9, 'systemPrompt',
        ha='left', va='center', color=BLUE, fontsize=10.5, family='monospace', fontweight='bold')
ax.text(right_x + 1.95, decl_y - 0.9, '시스템 프롬프트',
        ha='left', va='center', color=TEXT, fontsize=9.5)
ax.text(right_x + 0.35, decl_y - 1.2, 'tools',
        ha='left', va='center', color=BLUE, fontsize=10.5, family='monospace', fontweight='bold')
ax.text(right_x + 1.5, decl_y - 1.2, '사용할 도구 목록',
        ha='left', va='center', color=TEXT, fontsize=9.5)

# Arrow: declaration -> AgentCore (offset left, label offset right so they don't overlap)
arrow_x = right_x + col_w/2 - 1.3
ax.annotate('', xy=(arrow_x, 3.35), xytext=(arrow_x, 3.75),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
ax.text(arrow_x + 0.35, 3.55, 'AWS가 책임',
        ha='left', va='center', color=GREEN, fontsize=9, fontweight='bold')

# AWS-managed box
managed_y = 3.3
managed_h = 2.5
managed = FancyBboxPatch((right_x + 0.1, managed_y - managed_h), col_w - 0.4, managed_h,
                        boxstyle='round,pad=0.02,rounding_size=0.08',
                        linewidth=1.5, edgecolor=BLUE, facecolor=PANEL)
ax.add_patch(managed)
ax.text(right_x + 0.35, managed_y - 0.3, 'AgentCore가 자동 처리',
        ha='left', va='center', color=BLUE, fontsize=10, fontweight='bold')

managed_items = [
    '• Strands Agents 기반 agent loop',
    '• microVM 세션 격리 + 영속 파일시스템',
    '• IAM / SigV4 / 토큰 관리',
    '• Browser, Code Interpreter, Gateway 기본 도구',
    '• CDK 스택 생성·배포·Observability',
]
for i, item in enumerate(managed_items):
    ax.text(right_x + 0.4, managed_y - 0.65 - i * 0.35, item,
            ha='left', va='center', color=TEXT, fontsize=9)

# Bottom summary bar (right)
ax.text(right_x + col_w/2 - 0.3, 0.6, '3 API 호출 = 작동하는 에이전트',
        ha='center', va='center', color=GREEN, fontsize=11, fontweight='bold')
ax.text(right_x + col_w/2 - 0.3, 0.25, 'Preview · us-west-2, us-east-1, eu-central-1, ap-southeast-2',
        ha='center', va='center', color=SUBTEXT, fontsize=8)

# Left bottom summary
ax.text(left_x + col_w/2, 0.6, '6개 계층 × 수십 개 결정 포인트',
        ha='center', va='center', color=RED, fontsize=11, fontweight='bold')
ax.text(left_x + col_w/2, 0.25, '프레임워크 선택, 스토리지, IaC, 인증 설계까지',
        ha='center', va='center', color=SUBTEXT, fontsize=8)

# Watermark
fig.text(0.99, 0.012, 'jesamkim.github.io',
         ha='right', va='bottom', color='#6c7086',
         fontsize=8, alpha=0.7, fontstyle='italic')

plt.tight_layout()
out_path = '/Workshop/yan/ai-tech-blog/static/images/bedrock-agentcore-managed-harness-deep-dive/stack-comparison.png'
plt.savefig(out_path, dpi=150, facecolor=BG, bbox_inches='tight')
print(f'Saved: {out_path}')
