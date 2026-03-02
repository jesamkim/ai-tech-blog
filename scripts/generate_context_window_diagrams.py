#!/usr/bin/env python3
"""
Context Window 블로그 포스트용 다이어그램 생성
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 다크 테마 설정
plt.style.use('dark_background')
COLORS = {
    'bg': '#1a1a2e',
    'text': '#e0e0e0',
    'blue': '#4fc3f7',
    'green': '#66bb6a',
    'red': '#ef5350',
    'purple': '#ab47bc',
    'yellow': '#ffd54f'
}

def add_watermark(ax, fig):
    """워터마크 추가 (좌하단, 반투명 4%, min 14px max 28px)"""
    # 폰트 크기는 figure size에 따라 조정 (14-28px 범위)
    fig_width = fig.get_figwidth()
    fontsize = max(14, min(28, fig_width * 1.8))

    ax.text(0.02, 0.02, 'jesamkim.github.io',
            transform=ax.transAxes,
            fontsize=fontsize,
            color='white',
            alpha=0.04,
            ha='left',
            va='bottom',
            family='monospace')

def diagram_lost_in_middle():
    """Lost in the Middle U자형 성능 곡선"""
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    # U자형 곡선 데이터
    positions = np.linspace(0, 100, 50)
    # U자형: 시작과 끝이 높고 중간이 낮음
    performance = 85 - 25 * np.sin(np.pi * positions / 100)**2

    ax.plot(positions, performance, color=COLORS['blue'], linewidth=3, label='정확도')
    ax.fill_between(positions, performance, alpha=0.3, color=COLORS['blue'])

    # 주요 포인트 표시
    ax.scatter([0, 50, 100], [85, 60, 85], color=COLORS['yellow'], s=150, zorder=5, edgecolors='white', linewidths=2)
    ax.annotate('시작 부분\n(85%)', xy=(0, 85), xytext=(15, 75),
                fontsize=11, color=COLORS['text'],
                arrowprops=dict(arrowstyle='->', color=COLORS['yellow'], lw=2))
    ax.annotate('중간 부분\n(60%)', xy=(50, 60), xytext=(50, 45),
                fontsize=11, color=COLORS['text'], ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['yellow'], lw=2))
    ax.annotate('끝 부분\n(85%)', xy=(100, 85), xytext=(85, 75),
                fontsize=11, color=COLORS['text'],
                arrowprops=dict(arrowstyle='->', color=COLORS['yellow'], lw=2))

    ax.set_xlabel('문서 내 정보 위치 (%)', fontsize=12, color=COLORS['text'])
    ax.set_ylabel('검색 정확도 (%)', fontsize=12, color=COLORS['text'])
    ax.set_title('Lost in the Middle: 컨텍스트 중간에서 성능 저하',
                 fontsize=14, fontweight='bold', color=COLORS['text'], pad=20)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_ylim(40, 95)
    ax.tick_params(colors=COLORS['text'])

    add_watermark(ax, fig)
    plt.tight_layout()
    plt.savefig('/Workshop/yan/ai-tech-blog/static/images/posts/2026-03-02-context-window/lost-in-middle.png',
                dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print("✓ lost-in-middle.png 생성 완료")

def diagram_decision_flowchart():
    """RAG vs Long Context 의사결정 플로우차트"""
    fig, ax = plt.subplots(figsize=(12, 10), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 박스 스타일
    def draw_box(x, y, w, h, text, color, is_decision=False):
        if is_decision:
            # 다이아몬드 (의사결정)
            points = np.array([[x+w/2, y+h], [x+w, y+h/2], [x+w/2, y], [x, y+h/2]])
            polygon = mpatches.Polygon(points, facecolor=color, edgecolor='white', linewidth=2)
            ax.add_patch(polygon)
        else:
            # 둥근 사각형
            box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                facecolor=color, edgecolor='white', linewidth=2)
            ax.add_patch(box)

        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=10, color='white', weight='bold', wrap=True)

    def draw_arrow(x1, y1, x2, y2, label=''):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=20,
                              color='white', linewidth=2)
        ax.add_patch(arrow)
        if label:
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            ax.text(mid_x+0.3, mid_y, label, fontsize=9, color=COLORS['yellow'],
                   bbox=dict(boxstyle='round', facecolor=COLORS['bg'], alpha=0.8))

    # 플로우차트 구성
    draw_box(3.5, 10.5, 3, 1, '작업 시작', COLORS['blue'])
    draw_arrow(5, 10.5, 5, 9.5)

    draw_box(3.5, 8.5, 3, 1, '문서 크기 확인', COLORS['purple'], is_decision=True)
    draw_arrow(5, 8.5, 5, 7.5)

    # 200K 이하
    draw_box(0.5, 6.5, 2.5, 1, '< 200K\n토큰?', COLORS['green'])
    draw_arrow(3.5, 9, 2, 7.5, 'YES')
    draw_arrow(2, 6.5, 2, 5.5)
    draw_box(0.5, 4.5, 2.5, 1, 'Long Context\n직접 사용', COLORS['green'])

    # 200K~1M
    draw_box(3.5, 6.5, 3, 1, '200K~1M\n토큰?', COLORS['yellow'])
    draw_arrow(5, 7.5, 5, 6.5, 'NO')
    draw_arrow(5, 6.5, 5, 5.5)
    draw_box(3.5, 4.5, 3, 1, '추론 필요?', COLORS['purple'], is_decision=True)

    draw_arrow(3.5, 5, 2, 4, 'YES')
    draw_box(0.5, 3, 2.5, 0.8, 'Long Context\n(Caching 필수)', COLORS['blue'])

    draw_arrow(6.5, 5, 8, 4, 'NO')
    draw_box(7, 3, 2.5, 0.8, 'RAG\n(비용 절감)', COLORS['green'])

    # 1M 초과
    draw_box(7, 6.5, 2.5, 1, '> 1M\n토큰?', COLORS['red'])
    draw_arrow(6.5, 9, 8.5, 7.5, 'NO')
    draw_arrow(8.5, 6.5, 8.5, 5.5)
    draw_box(7, 4.5, 2.5, 1, 'RAG + 청킹\n(필수)', COLORS['red'])

    # 최종 단계
    draw_arrow(2, 3, 5, 2)
    draw_arrow(5, 3.8, 5, 2)
    draw_arrow(8.5, 3.8, 5, 2)
    draw_box(3.5, 0.5, 3, 1, '구현 완료', COLORS['blue'])

    ax.set_title('RAG vs Long Context 의사결정 플로우',
                 fontsize=16, fontweight='bold', color=COLORS['text'], pad=20)

    add_watermark(ax, fig)
    plt.tight_layout()
    plt.savefig('/Workshop/yan/ai-tech-blog/static/images/posts/2026-03-02-context-window/decision-flowchart.png',
                dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print("✓ decision-flowchart.png 생성 완료")

def diagram_cost_comparison():
    """비용 비교 차트"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=COLORS['bg'])

    # 왼쪽: 토큰당 비용
    ax1.set_facecolor(COLORS['bg'])
    contexts = ['50K', '100K', '200K', '500K', '1M']
    input_costs = [0.003, 0.003, 0.003, 0.006, 0.006]  # per 1K tokens
    output_costs = [0.015, 0.015, 0.015, 0.030, 0.030]

    x = np.arange(len(contexts))
    width = 0.35

    bars1 = ax1.bar(x - width/2, input_costs, width, label='Input', color=COLORS['blue'])
    bars2 = ax1.bar(x + width/2, output_costs, width, label='Output', color=COLORS['red'])

    ax1.set_xlabel('Context Window 크기', fontsize=12, color=COLORS['text'])
    ax1.set_ylabel('비용 (USD per 1K tokens)', fontsize=12, color=COLORS['text'])
    ax1.set_title('Claude Sonnet 4.6 토큰당 비용', fontsize=13, fontweight='bold', color=COLORS['text'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(contexts)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax1.tick_params(colors=COLORS['text'])

    # 200K 경계선
    ax1.axvline(x=1.5, color=COLORS['yellow'], linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(1.5, 0.028, '200K 경계', rotation=90, va='bottom', ha='right',
            color=COLORS['yellow'], fontsize=10, weight='bold')

    # 오른쪽: 레이턴시 비교
    ax2.set_facecolor(COLORS['bg'])
    scenarios = ['200K\n(No Cache)', '200K\n(Cached)', '1M\n(No Cache)', '1M\n(Cached)']
    prefill_times = [45, 7, 120, 18]  # seconds

    bars = ax2.bar(scenarios, prefill_times, color=[COLORS['red'], COLORS['green'],
                                                     COLORS['red'], COLORS['green']])

    # 값 표시
    for i, (bar, time) in enumerate(zip(bars, prefill_times)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{time}초\n({int(100-time/prefill_times[i//2*2]*100)}% ↓)' if i%2==1 else f'{time}초',
                ha='center', va='bottom', color=COLORS['text'], fontsize=10, weight='bold')

    ax2.set_ylabel('Prefill 시간 (초)', fontsize=12, color=COLORS['text'])
    ax2.set_title('Prompt Caching 레이턴시 절감 효과', fontsize=13, fontweight='bold', color=COLORS['text'])
    ax2.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax2.tick_params(colors=COLORS['text'])

    add_watermark(ax1, fig)
    plt.tight_layout()
    plt.savefig('/Workshop/yan/ai-tech-blog/static/images/posts/2026-03-02-context-window/cost-comparison.png',
                dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print("✓ cost-comparison.png 생성 완료")

def diagram_scenario_matrix():
    """사용 시나리오별 권장 전략 매트릭스"""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])
    ax.axis('off')

    # 테이블 데이터
    scenarios = [
        ['시나리오', '문서 크기', '추론 요구', '권장 전략', '비고'],
        ['코드 리뷰', '< 100K', '높음', 'Long Context', 'Caching 불필요'],
        ['기술 문서 QA', '100-200K', '중간', 'Long Context', 'Caching 권장'],
        ['계약서 분석', '200-500K', '높음', 'Long Context +\nCaching', '85% 레이턴시 절감'],
        ['법률 판례 검색', '500K-1M', '낮음', 'RAG', '비용 효율적'],
        ['대규모 코드베이스', '500K-1M', '높음', 'Long Context +\nCaching', 'LaRA 패턴'],
        ['기업 문서 검색', '> 1M', '낮음', 'RAG + 청킹', '필수'],
        ['멀티 리포지토리', '> 1M', '높음', 'RAG + Long Context\n하이브리드', 'U-NIAH 패턴'],
    ]

    # 색상 매핑
    strategy_colors = {
        'Long Context': COLORS['blue'],
        'Long Context +\nCaching': COLORS['green'],
        'RAG': COLORS['yellow'],
        'RAG + 청킹': COLORS['red'],
        'Long Context\n(No Cache)': COLORS['blue'],
        'RAG + Long Context\n하이브리드': COLORS['purple']
    }

    # 테이블 생성
    table = ax.table(cellText=scenarios, cellLoc='left',
                     loc='center', bbox=[0, 0, 1, 1])

    # 스타일링
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for i, row in enumerate(scenarios):
        for j, cell in enumerate(row):
            table_cell = table[(i, j)]

            # 헤더 행
            if i == 0:
                table_cell.set_facecolor(COLORS['purple'])
                table_cell.set_text_props(weight='bold', color='white', fontsize=11)
                table_cell.set_height(0.08)
            else:
                # 데이터 행
                if j == 3:  # 권장 전략 열
                    strategy = cell
                    color = strategy_colors.get(strategy, COLORS['bg'])
                    table_cell.set_facecolor(color)
                    table_cell.set_text_props(weight='bold', color='white')
                else:
                    table_cell.set_facecolor('#2a2a3e')
                    table_cell.set_text_props(color=COLORS['text'])

                table_cell.set_height(0.12)

            table_cell.set_edgecolor('white')
            table_cell.set_linewidth(1.5)

    # 열 너비 조정
    table.auto_set_column_width([0, 1, 2, 3, 4])
    for i in range(len(scenarios)):
        table[(i, 0)].set_width(0.15)  # 시나리오
        table[(i, 1)].set_width(0.12)  # 문서 크기
        table[(i, 2)].set_width(0.10)  # 추론 요구
        table[(i, 3)].set_width(0.20)  # 권장 전략
        table[(i, 4)].set_width(0.25)  # 비고

    ax.set_title('컨텍스트 윈도우 사용 시나리오별 권장 전략',
                 fontsize=16, fontweight='bold', color=COLORS['text'],
                 pad=20, y=0.98)

    # 범례
    legend_elements = [
        mpatches.Patch(color=COLORS['blue'], label='Long Context'),
        mpatches.Patch(color=COLORS['green'], label='Long Context + Caching'),
        mpatches.Patch(color=COLORS['yellow'], label='RAG'),
        mpatches.Patch(color=COLORS['red'], label='RAG + 청킹'),
        mpatches.Patch(color=COLORS['purple'], label='하이브리드')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.02),
             ncol=5, frameon=True, facecolor=COLORS['bg'], edgecolor='white')

    add_watermark(ax, fig)
    plt.tight_layout()
    plt.savefig('/Workshop/yan/ai-tech-blog/static/images/posts/2026-03-02-context-window/scenario-matrix.png',
                dpi=150, facecolor=COLORS['bg'])
    plt.close()
    print("✓ scenario-matrix.png 생성 완료")

if __name__ == '__main__':
    print("Context Window 다이어그램 생성 중...")
    diagram_lost_in_middle()
    diagram_decision_flowchart()
    diagram_cost_comparison()
    diagram_scenario_matrix()
    print("\n모든 다이어그램 생성 완료!")
