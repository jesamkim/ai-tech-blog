"""AgentCore blog diagrams redrawn in 3 themes (minimal / vibrant / editorial).

Preserves the structure and content of the three originals:
- diagram_agentcore_decision.py  → decision-tree
- diagram_agentcore_stack.py     → stack-comparison
- diagram_agentcore_session.py   → session-lifecycle

Outputs 9 PNGs to:
  /Workshop/yan/ai-tech-blog/static/images/agentcore-redraw/{theme}-{name}.png
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blog_diagram_theme import (  # noqa: E402
    THEMES,
    _get_theme,  # for access to palette colors when we need matching arrows/labels
    add_annotation,
    add_title,
    draw_arrow,
    draw_box,
    setup_figure,
)

OUT_DIR = '/Workshop/yan/ai-tech-blog/static/images/agentcore-redraw'
os.makedirs(OUT_DIR, exist_ok=True)


def _variant_line_color(theme: str, variant: str) -> str:
    """Fetch the line/edge color of a variant in a theme."""
    return _get_theme(theme)['palette'][variant]['line']


# ---------------------------------------------------------------------------
# 1. Decision tree
# ---------------------------------------------------------------------------

def diagram_decision_tree(theme: str) -> str:
    fig, ax = setup_figure(theme, figsize=(14, 8.5), xlim=(0, 14), ylim=(0, 9))
    add_title(
        ax,
        '어떤 방식을 선택할 것인가',
        'Managed Harness vs Code-defined Agent vs Bedrock Agents',
        theme=theme,
    )

    # Start node
    draw_box(
        ax, 5.5, 7.1, 3.0, 0.6,
        theme=theme, variant='primary',
        title='에이전트가 필요하다',
    )

    # Q1
    draw_box(
        ax, 4.5, 5.9, 5.0, 0.7,
        theme=theme, variant='accent',
        title='Q1. agent loop를 직접 구현해야 하는 제약이 있는가?',
        subtitle='(온프레미스, 특정 프레임워크, 규제 등)',
    )
    draw_arrow(ax, (7.0, 7.1), (7.0, 6.6),
               theme=theme, style='straight')

    # Q2
    draw_box(
        ax, 4.5, 4.4, 5.0, 0.7,
        theme=theme, variant='accent',
        title='Q2. 선언적 설정만으로 충분한가?',
        subtitle='(model + systemPrompt + tools 조합으로 커버)',
    )

    # Q1 → Q2 (NO path straight down)
    draw_arrow(ax, (7.0, 5.9), (7.0, 5.1),
               theme=theme, style='straight',
               color=_variant_line_color(theme, 'success'),
               label='NO', label_offset=(0.25, 0))

    # Outcome row
    box_h = 2.3
    box_w = 3.8

    # Code-defined (left, danger)
    draw_box(
        ax, 0.5, 1.0, box_w, box_h,
        theme=theme, variant='danger',
        title='Code-defined Agent',
        subtitle='Strands SDK 등으로 직접 구현',
        body=[
            '· Python 코드 풀 컨트롤',
            '· 복잡한 분기·루프 자유롭게',
            '· 온프레미스·하이브리드 가능',
            '· 디버깅 도구 선택 자유',
        ],
        title_align='left',
    )

    # Managed Harness (center, success)
    draw_box(
        ax, 5.1, 1.0, box_w, box_h,
        theme=theme, variant='success',
        title='Managed Harness',
        subtitle='3 선언 + CLI 배포 — Preview',
        body=[
            '· microVM 세션·영속 FS 무료 제공',
            '· CDK 배포 자동 생성',
            '· 모델·도구·프롬프트 설정 변경',
            '· Strands Agents 기반 동작',
        ],
        title_align='left',
    )

    # Bedrock Agents (right, secondary)
    draw_box(
        ax, 9.7, 1.0, box_w, box_h,
        theme=theme, variant='secondary',
        title='Bedrock Agents',
        subtitle='action groups + KB 기반',
        body=[
            '· OpenAPI action groups 중심',
            '· Knowledge Bases 내장 연동',
            '· Guardrails·감사로그 관리형',
            '· 엔터프라이즈 워크플로우',
        ],
        title_align='left',
    )

    # Q1 YES → Code-defined (curved arrow to top-center of code box)
    draw_arrow(
        ax, (4.9, 5.9), (0.5 + box_w / 2, 3.3),
        theme=theme, style='curved',
        color=_variant_line_color(theme, 'danger'),
        label='YES', label_offset=(-1.4, 0.3),
    )

    # Q2 YES → Managed Harness (straight down to top of MH)
    draw_arrow(
        ax, (7.0, 4.4), (5.1 + box_w / 2, 3.3),
        theme=theme, style='straight',
        color=_variant_line_color(theme, 'success'),
        label='YES', label_offset=(0.25, 0.15),
    )

    # Q2 NO → Bedrock Agents
    draw_arrow(
        ax, (9.1, 4.4), (9.7 + box_w / 2, 3.3),
        theme=theme, style='curved',
        color=_variant_line_color(theme, 'secondary'),
        label='복잡 workflow', label_offset=(0.35, 0.35),
    )

    add_annotation(
        ax, 7.0, 0.45,
        '선택 기준: 제어 수준 · 구현 난이도 · 운영 책임 범위',
        theme=theme, position='center',
    )

    out = os.path.join(OUT_DIR, f'{theme}-decision-tree.png')
    plt.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Stack comparison
# ---------------------------------------------------------------------------

def diagram_stack_comparison(theme: str) -> str:
    fig, ax = setup_figure(theme, figsize=(14, 8), xlim=(0, 14), ylim=(0, 8))
    add_title(
        ax,
        '에이전트 개발 스택 비교',
        'Traditional Stack vs Managed Harness',
        theme=theme,
    )

    left_x = 0.3
    right_x = 7.5
    col_w = 6.2

    # Left header
    add_annotation(
        ax, left_x + col_w / 2, 6.55,
        '기존 방식 (Traditional)',
        theme=theme, position='center',
        color=_variant_line_color(theme, 'danger'),
        weight='bold',
    )
    add_annotation(
        ax, left_x + col_w / 2, 6.2,
        '직접 구성하고 배포 스크립트까지 작성',
        theme=theme, position='center',
    )

    # Six layers on the left. Map semantic colors to theme variants.
    traditional_layers = [
        ('애플리케이션 코드', 'Agent loop, 프롬프트 튜닝', 'primary'),
        ('오케스트레이션 프레임워크', 'LangChain / LlamaIndex / Custom', 'secondary'),
        ('세션·메모리 계층', 'Redis, DynamoDB, 상태 직렬화', 'success'),
        ('도구 통합', 'Code execution, Browser, API wrappers', 'accent'),
        ('인증·식별자', 'IAM, Cognito, API keys, SigV4', 'success'),
        ('런타임·배포', 'ECS/EKS/Lambda + IaC + 모니터링', 'danger'),
    ]

    y_top = 5.6
    h = 0.7
    gap = 0.12
    for i, (title, detail, variant) in enumerate(traditional_layers):
        y_bottom = y_top - (i + 1) * h - i * gap
        draw_box(
            ax, left_x + 0.1, y_bottom, col_w - 0.2, h,
            theme=theme, variant=variant,
            title=title, subtitle=detail,
            title_align='left',
        )
        # Layer index at the right
        ax.text(
            left_x + col_w - 0.3, y_bottom + h / 2, f'계층 {i + 1}',
            ha='right', va='center',
            color=_variant_line_color(theme, variant),
            fontsize=8, style='italic',
        )

    # Divider
    spec = _get_theme(theme)
    divider_color = spec['text']['muted']
    ax.plot([7.2, 7.2], [0.5, 6.4], color=divider_color,
            linewidth=1, linestyle='--', alpha=0.6)

    # Right header
    add_annotation(
        ax, right_x + col_w / 2 - 0.3, 6.55,
        'Managed Harness',
        theme=theme, position='center',
        color=_variant_line_color(theme, 'success'),
        weight='bold',
    )
    add_annotation(
        ax, right_x + col_w / 2 - 0.3, 6.2,
        '3개의 선언만 있으면 배포 완료',
        theme=theme, position='center',
    )

    # Declaration box (top right)
    draw_box(
        ax, right_x + 0.1, 3.8, col_w - 0.4, 1.5,
        theme=theme, variant='success',
        title='개발자가 선언하는 것',
        body=[
            'model — 어떤 모델을 쓸지',
            'systemPrompt — 시스템 프롬프트',
            'tools — 사용할 도구 목록',
        ],
        title_align='left',
    )

    # Arrow: declaration → AgentCore-managed
    arrow_x = right_x + col_w / 2 - 1.3
    draw_arrow(
        ax, (arrow_x, 3.8), (arrow_x, 3.35),
        theme=theme, style='thick',
        color=_variant_line_color(theme, 'success'),
        label='AWS가 책임', label_offset=(0.35, 0),
    )

    # AWS-managed box
    draw_box(
        ax, right_x + 0.1, 0.8, col_w - 0.4, 2.5,
        theme=theme, variant='primary',
        title='AgentCore가 자동 처리',
        body=[
            '· Strands Agents 기반 agent loop',
            '· microVM 세션 격리 + 영속 파일시스템',
            '· IAM / SigV4 / 토큰 관리',
            '· Browser, Code Interpreter, Gateway 기본 도구',
            '· CDK 스택 생성·배포·Observability',
        ],
        title_align='left',
    )

    # Bottom summaries
    add_annotation(
        ax, left_x + col_w / 2, 0.55,
        '6개 계층 × 수십 개 결정 포인트',
        theme=theme, position='center',
        color=_variant_line_color(theme, 'danger'),
        weight='bold',
    )
    add_annotation(
        ax, left_x + col_w / 2, 0.25,
        '프레임워크 선택, 스토리지, IaC, 인증 설계까지',
        theme=theme, position='center',
    )

    # Right bottom summary sits inside the managed box area — move to just below.
    # The managed box ends at y=0.8, so push summary text into the bottom gutter.
    add_annotation(
        ax, right_x + col_w / 2 - 0.3, 0.45,
        '3 API 호출 = 작동하는 에이전트 · Preview',
        theme=theme, position='center',
        color=_variant_line_color(theme, 'success'),
        weight='bold',
    )
    add_annotation(
        ax, right_x + col_w / 2 - 0.3, 0.15,
        'us-west-2 · us-east-1 · eu-central-1 · ap-southeast-2',
        theme=theme, position='center',
    )

    out = os.path.join(OUT_DIR, f'{theme}-stack-comparison.png')
    plt.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. Session lifecycle
# ---------------------------------------------------------------------------

def diagram_session_lifecycle(theme: str) -> str:
    fig, ax = setup_figure(theme, figsize=(14, 8.5), xlim=(0, 14), ylim=(0, 9))
    add_title(
        ax,
        '세션 생명주기 — microVM 격리와 영속 파일시스템',
        'Session Lifecycle: isolated compute, persistent state, suspend & resume',
        theme=theme,
    )

    # Client box at top
    draw_box(
        ax, 5.5, 6.8, 3.0, 0.7,
        theme=theme, variant='primary',
        title='클라이언트 / 애플리케이션',
        subtitle='agentcore invoke --session-id <uuid>',
    )

    # Arrow down to harness
    draw_arrow(
        ax, (7.0, 6.8), (7.0, 6.3),
        theme=theme, style='straight',
        color=_variant_line_color(theme, 'primary'),
        label='session_id 로 라우팅', label_offset=(0.25, 0),
    )

    # Harness
    draw_box(
        ax, 4.5, 5.6, 5.0, 0.7,
        theme=theme, variant='success',
        title='Managed Harness Runtime',
        subtitle='Strands Agents loop · 모델 호출 · 도구 라우팅',
    )

    # Three microVMs row
    vm_y_bottom = 3.7
    vm_h = 1.3
    vm_w = 3.5
    vms = [
        ('session_id = A', 'Active', 'success', 1.0),
        ('session_id = B', 'Suspended', 'accent', 5.0),
        ('session_id = C', 'Resumed', 'primary', 9.0),
    ]
    for (sid, state, variant, x0) in vms:
        draw_box(
            ax, x0, vm_y_bottom, vm_w, vm_h,
            theme=theme, variant=variant,
            title=f'microVM — {state}',
            subtitle=sid,
            body=[
                'dedicated CPU / memory',
                '/workspace shell + FS',
            ],
        )
        # Arrow from harness to this VM top
        draw_arrow(
            ax, (7.0, 5.6), (x0 + vm_w / 2, vm_y_bottom + vm_h),
            theme=theme, style='curved',
            color=_variant_line_color(theme, variant),
        )

    # Persistent filesystem band
    fs_y_bottom = 2.1
    fs_h = 0.9
    draw_box(
        ax, 1.0, fs_y_bottom, 12.0, fs_h,
        theme=theme, variant='secondary',
        title='Persistent Filesystem',
        subtitle='세션 상태·중간 결과·셸 히스토리 저장 · microVM이 종료되어도 유지 · resume 가능',
        title_align='left',
    )

    # Arrows from each VM down to FS
    for (_, _, variant, x0) in vms:
        draw_arrow(
            ax, (x0 + vm_w / 2, vm_y_bottom), (x0 + vm_w / 2, fs_y_bottom + fs_h),
            theme=theme, style='straight',
            color=_variant_line_color(theme, variant),
        )

    # Bottom scenario steps
    flow_y = 1.0
    add_annotation(
        ax, 7.0, flow_y + 0.55,
        '중단·재개 시나리오',
        theme=theme, position='center',
        color=_variant_line_color(theme, 'accent'),
        weight='bold',
    )

    steps = [
        ('1', '호출', '클라이언트가 session_id 로 invoke', 'primary', 1.2),
        ('2', '실행', 'microVM이 작업 수행·셸 명령 실행', 'success', 4.5),
        ('3', '중단', '유휴시 VM 정지, FS만 보존', 'accent', 7.8),
        ('4', '재개', '같은 session_id로 상태 이어받기', 'secondary', 11.1),
    ]
    for (num, title, detail, variant, x) in steps:
        color = _variant_line_color(theme, variant)
        spec = _get_theme(theme)
        # Filled circle with number
        circle = plt.Circle((x, flow_y), 0.15, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(x, flow_y, num, ha='center', va='center',
                color=spec['canvas'], fontsize=8, fontweight='bold', zorder=4)
        ax.text(x + 0.25, flow_y + 0.08, title, ha='left', va='center',
                color=color, fontsize=9, fontweight='bold', zorder=4)
        ax.text(x + 0.25, flow_y - 0.18, detail, ha='left', va='center',
                color=spec['text']['secondary'], fontsize=8, zorder=4)

    # Arrows between steps
    for i in range(len(steps) - 1):
        x1 = steps[i][4] + 2.6
        x2 = steps[i + 1][4] - 0.18
        draw_arrow(
            ax, (x1, flow_y), (x2, flow_y),
            theme=theme, style='straight',
        )

    out = os.path.join(OUT_DIR, f'{theme}-session-lifecycle.png')
    plt.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Orchestrate
# ---------------------------------------------------------------------------

def main() -> int:
    diagrams = {
        'decision-tree': diagram_decision_tree,
        'stack-comparison': diagram_stack_comparison,
        'session-lifecycle': diagram_session_lifecycle,
    }
    generated = []
    errors = []
    for theme in THEMES:
        for name, fn in diagrams.items():
            try:
                out = fn(theme)
                size_kb = os.path.getsize(out) / 1024
                generated.append((theme, name, out, size_kb))
                print(f'[ok] {theme}/{name} → {out} ({size_kb:.1f} KB)')
            except Exception as e:  # noqa: BLE001
                errors.append((theme, name, str(e)))
                print(f'[FAIL] {theme}/{name}: {e}')

    print(f'\nGenerated: {len(generated)} files in {OUT_DIR}')
    if errors:
        print(f'Errors: {len(errors)}')
        for t, n, e in errors:
            print(f'  {t}/{n}: {e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
