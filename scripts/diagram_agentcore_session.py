"""Session lifecycle: microVM + persistent FS + suspend/resume."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

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

ax.text(7, 8.45, '세션 생명주기 — microVM 격리와 영속 파일시스템',
        ha='center', va='center', color=TEXT, fontsize=17, fontweight='bold')
ax.text(7, 8.02, 'Session Lifecycle: isolated compute, persistent state, suspend & resume',
        ha='center', va='center', color=SUBTEXT, fontsize=10, style='italic')

# Client at top
client_box = FancyBboxPatch((5.5, 6.8), 3, 0.7,
                            boxstyle='round,pad=0.02,rounding_size=0.1',
                            linewidth=1.5, edgecolor=BLUE, facecolor=PANEL)
ax.add_patch(client_box)
ax.text(7, 7.27, '클라이언트 / 애플리케이션',
        ha='center', va='center', color=BLUE, fontsize=11, fontweight='bold')
ax.text(7, 6.97, 'agentcore invoke --session-id <uuid>',
        ha='center', va='center', color=TEXT, fontsize=8.5, family='monospace')

# Arrow down to harness
ax.annotate('', xy=(7, 6.4), xytext=(7, 6.75),
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5))
ax.text(7.2, 6.55, 'session_id 로 라우팅',
        ha='left', va='center', color=SUBTEXT, fontsize=8.5)

# Harness controller
harness_box = FancyBboxPatch((4.5, 5.6), 5, 0.7,
                             boxstyle='round,pad=0.02,rounding_size=0.1',
                             linewidth=1.5, edgecolor=GREEN, facecolor=PANEL)
ax.add_patch(harness_box)
ax.text(7, 6.07, 'Managed Harness Runtime',
        ha='center', va='center', color=GREEN, fontsize=11, fontweight='bold')
ax.text(7, 5.78, 'Strands Agents loop · 모델 호출 · 도구 라우팅',
        ha='center', va='center', color=SUBTEXT, fontsize=8.5)

# Three microVMs row
vm_y = 4.7
vm_h = 1.3
vm_w = 3.5
vm_gap = 0.4

vms = [
    ('session_id = A', 'Active', GREEN, 1.0),
    ('session_id = B', 'Suspended', YELLOW, 5.0),
    ('session_id = C', 'Resumed', BLUE, 9.0),
]

for (sid, state, color, x0) in vms:
    box = FancyBboxPatch((x0, vm_y - vm_h + 0.3), vm_w, vm_h,
                         boxstyle='round,pad=0.02,rounding_size=0.08',
                         linewidth=1.5, edgecolor=color, facecolor='#0d1b2a')
    ax.add_patch(box)
    ax.text(x0 + vm_w/2, vm_y + 0.15, 'microVM',
            ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    ax.text(x0 + vm_w/2, vm_y - 0.1, sid,
            ha='center', va='center', color=TEXT, fontsize=8.5, family='monospace')
    # State chip
    chip = FancyBboxPatch((x0 + vm_w/2 - 0.55, vm_y - 0.45), 1.1, 0.28,
                          boxstyle='round,pad=0.01,rounding_size=0.1',
                          linewidth=0.8, edgecolor=color, facecolor='none')
    ax.add_patch(chip)
    ax.text(x0 + vm_w/2, vm_y - 0.31, state,
            ha='center', va='center', color=color, fontsize=8)
    # Resources inside
    ax.text(x0 + 0.15, vm_y - 0.75, 'dedicated CPU / memory',
            ha='left', va='center', color=SUBTEXT, fontsize=8, family='monospace')
    ax.text(x0 + 0.15, vm_y - 0.93, '/workspace shell + FS',
            ha='left', va='center', color=SUBTEXT, fontsize=8, family='monospace')

# Arrows from harness to VMs
for (_, _, color, x0) in vms:
    ax.annotate('', xy=(x0 + vm_w/2, vm_y + 0.3), xytext=(7, 5.6),
                arrowprops=dict(arrowstyle='->', color=color, lw=1, alpha=0.6))

# Persistent filesystem band
fs_y = 2.5
fs_h = 1.0
fs_box = FancyBboxPatch((1.0, fs_y - fs_h + 0.2), 12, fs_h,
                        boxstyle='round,pad=0.02,rounding_size=0.1',
                        linewidth=1.5, edgecolor=PURPLE, facecolor=PANEL)
ax.add_patch(fs_box)
ax.text(1.3, fs_y + 0.05, 'Persistent Filesystem',
        ha='left', va='center', color=PURPLE, fontsize=11, fontweight='bold')
ax.text(1.3, fs_y - 0.25, '세션 상태·중간 결과·셸 히스토리 저장 · microVM이 종료되어도 유지',
        ha='left', va='center', color=SUBTEXT, fontsize=9)
ax.text(12.7, fs_y - 0.1, 'resume 가능',
        ha='right', va='center', color=PURPLE, fontsize=9, style='italic')

# Arrows from each VM down to FS
for (_, _, color, x0) in vms:
    ax.annotate('', xy=(x0 + vm_w/2, fs_y + 0.2), xytext=(x0 + vm_w/2, vm_y - vm_h + 0.32),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1, alpha=0.7))

# Bottom flow: suspend/resume scenario
flow_y = 1.1
ax.text(7, flow_y + 0.45, '중단·재개 시나리오',
        ha='center', va='center', color=YELLOW, fontsize=11, fontweight='bold')

steps = [
    ('1', '호출', '클라이언트가 session_id 로 invoke', BLUE, 1.2),
    ('2', '실행', 'microVM이 작업 수행·셸 명령 실행', GREEN, 4.5),
    ('3', '중단', '유휴시 VM 정지, FS만 보존', YELLOW, 7.8),
    ('4', '재개', '같은 session_id로 상태 이어받기', PURPLE, 11.1),
]
for (num, title, detail, color, x) in steps:
    circle = plt.Circle((x, flow_y), 0.12, color=color, zorder=3)
    ax.add_patch(circle)
    ax.text(x, flow_y, num, ha='center', va='center', color=BG, fontsize=8, fontweight='bold', zorder=4)
    ax.text(x + 0.2, flow_y + 0.05, title, ha='left', va='center', color=color, fontsize=9, fontweight='bold')
    ax.text(x + 0.2, flow_y - 0.18, detail, ha='left', va='center', color=SUBTEXT, fontsize=7.8)

# Arrows between steps
for i in range(3):
    x1 = steps[i][4] + 2.4
    x2 = steps[i+1][4] - 0.14
    ax.annotate('', xy=(x2, flow_y), xytext=(x1, flow_y),
                arrowprops=dict(arrowstyle='->', color=BORDER, lw=1))

# Watermark
fig.text(0.99, 0.012, 'jesamkim.github.io',
         ha='right', va='bottom', color='#6c7086',
         fontsize=8, alpha=0.7, fontstyle='italic')

plt.tight_layout()
out_path = '/Workshop/yan/ai-tech-blog/static/images/bedrock-agentcore-managed-harness-deep-dive/session-lifecycle.png'
plt.savefig(out_path, dpi=150, facecolor=BG, bbox_inches='tight')
print(f'Saved: {out_path}')
