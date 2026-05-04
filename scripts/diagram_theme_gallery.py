"""Generate 9 sample diagrams (3 themes x 3 scenarios) for the gallery.

Run from repo root::

    python3 scripts/diagram_theme_gallery.py

Outputs to ``static/images/theme-gallery/{theme}-{scenario}.png``.
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blog_diagram_theme import (  # noqa: E402
    THEMES,
    add_annotation,
    add_title,
    draw_arrow,
    draw_box,
    setup_figure,
)

OUT_DIR = '/Workshop/yan/ai-tech-blog/static/images/theme-gallery'
os.makedirs(OUT_DIR, exist_ok=True)

SCENARIOS = ('architecture', 'decision-tree', 'flowchart')


# ---------------------------------------------------------------------------
# Scenario A: 4-layer architecture
# ---------------------------------------------------------------------------

def scene_architecture(theme: str) -> str:
    fig, ax = setup_figure(theme, figsize=(14, 8), xlim=(0, 14), ylim=(0, 8))
    add_title(
        ax,
        '서비스 아키텍처 개요',
        'Client · API · Service · Storage 4계층',
        theme=theme,
    )

    # Four horizontal layers, stacked vertically
    layers = [
        ('Client Layer', '웹·모바일·CLI 진입점', 'primary',
         ['React SPA · iOS · Android · CLI']),
        ('API Layer', 'REST + WebSocket 라우팅', 'accent',
         ['API Gateway HTTP/v2 · Cognito + JWT 인증 · Rate Limit']),
        ('Service Layer', '비즈니스 로직 / Agent 실행', 'secondary',
         ['Strands Agent 런타임 · Bedrock Claude 호출 · microVM 세션 격리']),
        ('Storage Layer', '영속 데이터 / 세션 상태', 'success',
         ['DynamoDB session/state · S3 실행 로그 · OpenSearch 검색']),
    ]

    box_w = 12.0
    box_h = 1.2
    base_x = 1.0
    top_y = 6.7

    for i, (title, sub, variant, body) in enumerate(layers):
        y = top_y - i * (box_h + 0.3)
        draw_box(
            ax, base_x, y - box_h, box_w, box_h,
            theme=theme, variant=variant,
            title=title, subtitle=sub,
            body=body, title_align='left',
        )
        if i < len(layers) - 1:
            cx = base_x + box_w / 2
            draw_arrow(
                ax, (cx, y - box_h - 0.02),
                (cx, y - box_h - 0.28),
                theme=theme, style='thick',
            )

    add_annotation(
        ax, 7.0, 0.45,
        '흐름: 클라이언트 요청은 API를 거쳐 Agent가 처리하고, 결과·세션 상태를 Storage에 기록',
        theme=theme, position='center',
    )

    out = os.path.join(OUT_DIR, f'{theme}-architecture.png')
    plt.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Scenario B: Decision tree (Root → 2 questions → 4 outcomes)
# ---------------------------------------------------------------------------

def scene_decision(theme: str) -> str:
    fig, ax = setup_figure(theme, figsize=(14, 8.5), xlim=(0, 14), ylim=(0, 9))
    add_title(
        ax,
        '에이전트 배포 방식 의사결정',
        'Q1 제약 여부 · Q2 설정 복잡도',
        theme=theme,
    )

    # Root
    draw_box(ax, 5.5, 7.1, 3.0, 0.7,
             theme=theme, variant='primary',
             title='에이전트가 필요하다')

    # Q1
    draw_box(ax, 4.3, 5.8, 5.4, 0.9,
             theme=theme, variant='accent',
             title='Q1. 직접 구현 제약이 있는가?',
             subtitle='(온프레·규제·커스텀 프레임워크)')

    draw_arrow(ax, (7.0, 7.1), (7.0, 6.7),
               theme=theme, style='straight')

    # Q2
    draw_box(ax, 4.3, 4.2, 5.4, 0.9,
             theme=theme, variant='accent',
             title='Q2. 선언만으로 충분한가?',
             subtitle='(model · prompt · tools 조합)')

    draw_arrow(ax, (7.0, 5.8), (7.0, 5.1),
               theme=theme, style='straight',
               label='NO', label_offset=(0.25, 0))

    # Outcomes (3 columns wide at bottom)
    box_h = 2.3
    box_w = 3.8

    outcomes = [
        (0.4, 'danger', 'Code-defined', 'Strands SDK 직접 구현',
         ['Python 풀 컨트롤', '복잡 분기·루프 자유', '온프레·하이브리드 OK']),
        (5.1, 'success', 'Managed Harness', '3 선언 + CLI 배포',
         ['microVM 세션 제공', 'CDK 스택 자동 생성', '모델·툴 즉시 교체']),
        (9.8, 'secondary', 'Bedrock Agents', 'action group + KB',
         ['OpenAPI 기반', 'Knowledge Bases 내장', 'Guardrails 관리형']),
    ]

    for (x, variant, title, sub, pros) in outcomes:
        draw_box(ax, x, 1.0, box_w, box_h,
                 theme=theme, variant=variant,
                 title=title, subtitle=sub,
                 body=[f'· {p}' for p in pros],
                 title_align='left')

    # Branch arrows from Q1 to Code-defined, Q2 to MH, Q2 to BA
    draw_arrow(ax, (4.8, 5.8), (2.3, 3.3),
               theme=theme, style='curved',
               label='YES', label_offset=(-1.2, 0.3))
    draw_arrow(ax, (7.0, 4.2), (7.0, 3.3),
               theme=theme, style='straight',
               label='YES', label_offset=(0.25, 0))
    draw_arrow(ax, (9.3, 4.2), (11.7, 3.3),
               theme=theme, style='curved',
               label='복잡 workflow', label_offset=(0.3, 0.3))

    add_annotation(
        ax, 7.0, 0.4,
        '선택 기준 — 제어 수준 · 구현 난이도 · 운영 책임 범위',
        theme=theme, position='center',
    )

    out = os.path.join(OUT_DIR, f'{theme}-decision-tree.png')
    plt.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Scenario C: 5-step flowchart with 1 branch
# ---------------------------------------------------------------------------

def scene_flowchart(theme: str) -> str:
    fig, ax = setup_figure(theme, figsize=(14, 7.5), xlim=(0, 14), ylim=(0, 7.5))
    add_title(
        ax,
        '요청 처리 파이프라인',
        '5단계 선형 + 검증 실패 시 분기',
        theme=theme,
    )

    # Linear 5 steps on middle row
    steps = [
        ('Receive', '요청 수신', 'primary'),
        ('Validate', '스키마 검증', 'accent'),
        ('Enrich', '컨텍스트 결합', 'secondary'),
        ('Execute', '에이전트 실행', 'success'),
        ('Respond', '결과 반환', 'primary'),
    ]
    box_w = 2.1
    box_h = 1.2
    gap = 0.4
    total_w = len(steps) * box_w + (len(steps) - 1) * gap
    start_x = (14 - total_w) / 2
    mid_y = 4.2

    positions = []
    for i, (title, sub, variant) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        draw_box(
            ax, x, mid_y, box_w, box_h,
            theme=theme, variant=variant,
            title=title, subtitle=sub,
        )
        positions.append((x, x + box_w))

    # Arrows between adjacent boxes
    for i in range(len(steps) - 1):
        right = positions[i][1]
        nxt_left = positions[i + 1][0]
        y = mid_y + box_h / 2
        draw_arrow(ax, (right + 0.02, y), (nxt_left - 0.02, y),
                   theme=theme, style='straight')

    # Branch: from Validate down to Error/Return
    v_x = (positions[1][0] + positions[1][1]) / 2
    draw_arrow(ax, (v_x, mid_y), (v_x, 2.6),
               theme=theme, style='dashed',
               label='invalid', label_offset=(0.15, 0.2))

    draw_box(
        ax, v_x - 1.6, 1.3, 3.2, 1.3,
        theme=theme, variant='danger',
        title='400 Bad Request', subtitle='검증 실패 시 즉시 응답',
        body=['에러 메시지 + field 위치', '메트릭 invalid_request_count'],
        title_align='left',
    )

    # Annotation top-left and right
    add_annotation(
        ax, start_x, mid_y + box_h + 0.35,
        '정상 경로 — 좌→우 순차 처리',
        theme=theme, position='left',
    )
    add_annotation(
        ax, 13.5, mid_y + box_h + 0.35,
        '평균 p50 < 250ms',
        theme=theme, position='right',
    )

    out = os.path.join(OUT_DIR, f'{theme}-flowchart.png')
    plt.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Orchestrate
# ---------------------------------------------------------------------------

def main() -> int:
    scenarios = {
        'architecture': scene_architecture,
        'decision-tree': scene_decision,
        'flowchart': scene_flowchart,
    }
    generated = []
    errors = []
    for theme in THEMES:
        for name, fn in scenarios.items():
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
        for t, s, e in errors:
            print(f'  {t}/{s}: {e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
