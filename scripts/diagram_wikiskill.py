#!/usr/bin/env python3
"""WikiSkill 포스트용 이미지 3종 생성 (light vercel-stripe 스타일).

출력: static/images/wikiskill-persistent-knowledge-skill-evolution/ 아래
  - wikiskill-architecture.svg / .png   (DIAGRAM 1: 3계층 + 4구성요소 데이터 흐름)
  - wikiskill-ablation.svg / .png        (DIAGRAM 2: Table 3 ablation 막대 그래프)
  - wikiskill-cover.svg / .png           (COVER: 저장소 3개 개념도)

모든 수치는 arXiv:2608.27454v1 Table 3 원문 값.
SVG는 자기완결형(폰트는 시스템 NanumSquareRound / Noto Sans CJK KR 사용).
PNG 변환은 별도 rsvg-convert 호출.
"""
import html
from pathlib import Path

OUT = Path("static/images/wikiskill-persistent-knowledge-skill-evolution")
OUT.mkdir(parents=True, exist_ok=True)

# ── light vercel-stripe 팔레트 ────────────────────────────────
BG        = "#ffffff"
PANEL     = "#f6f8fa"   # 옅은 회색 패널
BORDER    = "#e2e8f0"   # 얇은 회색 보더
INK       = "#0f172a"   # 진한 잉크 (제목)
SUB       = "#475569"   # 서브 텍스트
MUTE      = "#94a3b8"   # 흐린 텍스트
BLUE      = "#3b82f6"
INDIGO    = "#6366f1"
VIOLET    = "#8b5cf6"
GREEN     = "#10b981"
AMBER     = "#f59e0b"
ROSE      = "#f43f5e"
RAW_C     = "#64748b"   # raw: 회색 계열
WIKI_C    = "#6366f1"   # wiki: 인디고
SKILL_C   = "#10b981"   # skills: 그린
FONT      = "'NanumSquareRound','Noto Sans CJK KR',sans-serif"

WM = "jesamkim.github.io"


def esc(s):
    return html.escape(s, quote=True)


def watermark(w, h):
    """좌하단 워터마크."""
    return (f'<text x="16" y="{h-14}" font-family="{FONT}" font-size="13" '
            f'font-weight="700" fill="{MUTE}" opacity="0.55">{WM}</text>')


def defs():
    return (
        '<defs>'
        f'<linearGradient id="gTitle" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{BLUE}"/><stop offset="0.5" stop-color="{INDIGO}"/>'
        f'<stop offset="1" stop-color="{VIOLET}"/></linearGradient>'
        f'<linearGradient id="gCover" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="#eef2ff"/><stop offset="1" stop-color="#faf5ff"/></linearGradient>'
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{SUB}"/></marker>'
        f'<marker id="arrowG" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{SKILL_C}"/></marker>'
        f'<marker id="arrowR" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{ROSE}"/></marker>'
        '</defs>'
    )


def card(x, y, w, h, fill=PANEL, stroke=BORDER, rx=14, sw=1.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=15, fill=INK, weight="400", anchor="start", spacing=None):
    sp = f' letter-spacing="{spacing}"' if spacing else ''
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}"{sp}>{esc(s)}</text>')


# ══════════════════════════════════════════════════════════════
# DIAGRAM 1 : 3계층 + 4구성요소 데이터 흐름
# ══════════════════════════════════════════════════════════════
def build_architecture():
    W, H = 1120, 820
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    s.append(defs())
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

    # 제목
    s.append(text(40, 52, "WikiSkill 진화 루프: 세 저장소와 네 구성 요소", 26, "url(#gTitle)", "800"))
    s.append(text(40, 80, "스킬은 되돌리고 지식(wiki)은 되돌리지 않는 비대칭 구조", 15, SUB, "500"))

    # 세 계층 패널 (세로 밴드)
    band_y, band_h = 118, 480
    cols = [
        (40,  "raw/",    "원본 실행 trace",   "immutable (고치지 않음)",   RAW_C),
        (410, "wiki/",   "누적 지식 저장소",   "반복 사이 초기화 안 됨",     WIKI_C),
        (780, "skills/", "활성 스킬",         "Inference Agent가 읽음",    SKILL_C),
    ]
    band_w = 300
    for x, name, desc, note, col in cols:
        s.append(card(x, band_y, band_w, band_h, fill="#ffffff", stroke=BORDER, rx=16, sw=1.5))
        s.append(f'<rect x="{x}" y="{band_y}" width="{band_w}" height="6" rx="3" fill="{col}"/>')
        s.append(text(x+22, band_y+42, name, 22, col, "800"))
        s.append(text(x+22, band_y+68, desc, 14, INK, "600"))
        s.append(text(x+22, band_y+90, note, 12.5, MUTE, "500"))

    # raw/ 내부 항목
    rx0 = 62
    for i, t in enumerate(["추론 과정", "tool 호출", "tool 출력", "최종 답변"]):
        yy = band_y + 128 + i*82
        s.append(card(rx0, yy, 256, 62, fill=PANEL, stroke=BORDER, rx=10, sw=1))
        s.append(text(rx0+18, yy+37, t, 15, SUB, "600"))

    # wiki/ 내부 항목
    wx0 = 432
    wiki_items = [
        ("patterns/", "실패 유형 · 성공 전략 페이지"),
        ("index.md", "pattern 목록"),
        ("logs.md", "Wiki Maintainer 진화 로그"),
        ("skill-impact.md", "제안 diff · 수락 여부 감사 기록"),
    ]
    for i, (a, b) in enumerate(wiki_items):
        yy = band_y + 128 + i*82
        s.append(card(wx0, yy, 256, 62, fill=PANEL, stroke=BORDER, rx=10, sw=1))
        s.append(text(wx0+16, yy+27, a, 14.5, WIKI_C, "700"))
        s.append(text(wx0+16, yy+48, b, 12, SUB, "500"))

    # skills/ 내부 항목
    sx0 = 802
    for i, (a, b) in enumerate([("SKILL.md", "스킬 본문"),
                                ("PURPOSE.md", "유발 wiki pattern 매핑")]):
        yy = band_y + 138 + i*168
        s.append(card(sx0, yy, 256, 120, fill=PANEL, stroke=BORDER, rx=10, sw=1))
        s.append(text(sx0+18, yy+40, a, 16, SKILL_C, "700"))
        s.append(text(sx0+18, yy+66, b, 13, SUB, "500"))

    # 구성요소 라벨(밴드 사이 흐름 화살표)
    def flow(x1, y, x2, label, col, marker):
        mid = (x1+x2)/2
        s.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{col}" '
                 f'stroke-width="2.2" marker-end="url(#{marker})"/>')
        s.append(f'<rect x="{mid-96}" y="{y-30}" width="192" height="26" rx="13" '
                 f'fill="#ffffff" stroke="{BORDER}" stroke-width="1"/>')
        s.append(text(mid, y-12, label, 12.5, col, "700", "middle"))

    # raw -> wiki : Wiki Maintainer / wiki -> skills : Skill Proposer
    flow(340, 250, 410, "Wiki Maintainer", WIKI_C, "arrow")
    flow(710, 250, 780, "Skill Proposer", SKILL_C, "arrow")

    # ── 하단 영역: Gating & Rollback + 루프 ──
    gb_y = band_y + band_h + 26   # 624
    # Gating & Rollback 설명 (밴드 아래, skills 열 아래에 배치)
    s.append(card(560, gb_y, 498, 96, fill="#fff7ed", stroke="#fed7aa", rx=12, sw=1.2))
    s.append(text(582, gb_y+30, "Gating & Rollback", 15, AMBER, "800"))
    s.append(text(582, gb_y+54, "validation 점수가 최고치를 넘을 때만 수락", 13, SUB, "500"))
    s.append(text(582, gb_y+76, "거부되면 skills만 롤백, wiki는 롤백하지 않음", 12.5, ROSE, "600"))

    # Inference Agent 라벨 + wiki 접근 차단 (밴드 아래 좌측)
    s.append(card(40, gb_y, 496, 96, fill="#eff6ff", stroke="#bfdbfe", rx=12, sw=1.2))
    s.append(text(62, gb_y+30, "Inference Agent 학습 rollout", 15, BLUE, "800"))
    s.append(text(62, gb_y+54, "활성 스킬을 system prompt에 직접 주입", 13, SUB, "500"))
    # wiki 접근 차단 뱃지
    s.append(f'<circle cx="74" cy="{gb_y+76}" r="9" fill="#fff" stroke="{ROSE}" stroke-width="2"/>')
    s.append(f'<line x1="68" y1="{gb_y+70}" x2="80" y2="{gb_y+82}" stroke="{ROSE}" stroke-width="2"/>')
    s.append(text(90, gb_y+80, "학습 rollout 중 wiki 접근 차단", 12.5, ROSE, "600"))

    # 루프 화살표: skills 하단 -> 아래 -> raw 하단 (닫힌 루프)
    loop_y = gb_y + 130   # 754
    s.append(f'<path d="M 930 {band_y+band_h} L 930 {loop_y} L 190 {loop_y} L 190 {band_y+band_h}" '
             f'fill="none" stroke="{BLUE}" stroke-width="2.2" '
             f'marker-end="url(#arrow)"/>')
    s.append(f'<rect x="454" y="{loop_y-14}" width="212" height="28" rx="14" '
             f'fill="#ffffff" stroke="{BLUE}" stroke-width="1.2"/>')
    s.append(text(560, loop_y+5, "새 trace로 다음 반복", 13, BLUE, "700", "middle"))

    s.append(watermark(W, H))
    s.append('</svg>')
    (OUT / "wikiskill-architecture.svg").write_text("".join(s), encoding="utf-8")


# ══════════════════════════════════════════════════════════════
# DIAGRAM 2 : Table 3 ablation 막대 그래프
# ══════════════════════════════════════════════════════════════
def build_ablation():
    W, H = 1120, 680
    # 원문 Table 3 (Gemini-3.5-Flash). ALFWorld 제외 4벤치 + 평균.
    benches = ["LiveMath", "SealQA", "SpreadSheet", "OfficeQA", "평균"]
    # 설정: (라벨, 색, [LiveMath, SealQA, SpreadSheet, OfficeQA, 평균])
    configs = [
        ("스킬 없음",                    MUTE,   [33.0, 29.4, 50.5, 48.6, 40.4]),
        ("Inf 허용 · Prop 차단",         "#cbd5e1", [43.8, 42.0, 44.4, 51.0, 45.3]),
        ("Inf 차단 · Prop 차단",         "#a5b4fc", [51.3, 38.4, 49.9, 55.2, 48.7]),
        ("Inf 허용 · Prop 허용",         VIOLET, [64.8, 42.8, 80.2, 55.6, 60.9]),
        ("Inf 차단 · Prop 허용 (기본)",  INDIGO, [72.6, 44.7, 76.6, 60.7, 63.7]),
    ]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    s.append(defs())
    s.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')
    s.append(text(40, 50, "wiki 접근 조건별 성능 (Table 3 · Gemini-3.5-Flash)", 25, "url(#gTitle)", "800"))
    s.append(text(40, 78, "Inference Agent와 Skill Proposer의 wiki 접근을 각각 켜고 끈 네 구성과 스킬 없음 기준선",
                  14, SUB, "500"))

    # 플롯 영역
    px, py = 90, 130
    pw, ph = W-130, 420
    base = py + ph
    ymax = 90.0
    # 격자 + y축 라벨
    for v in range(0, int(ymax)+1, 15):
        gy = base - (v/ymax)*ph
        s.append(f'<line x1="{px}" y1="{gy:.1f}" x2="{px+pw}" y2="{gy:.1f}" stroke="{BORDER}" stroke-width="1"/>')
        s.append(text(px-12, gy+4, str(v), 12, MUTE, "500", "end"))
    s.append(text(px-12, py-14, "정확도 (%)", 12.5, SUB, "600", "start"))

    n_groups = len(benches)
    n_bars = len(configs)
    group_w = pw / n_groups
    bar_gap = 6
    bar_w = (group_w - 40 - bar_gap*(n_bars-1)) / n_bars

    for gi, bench in enumerate(benches):
        gx = px + gi*group_w + 20
        for ci, (label, col, vals) in enumerate(configs):
            v = vals[gi]
            bh = (v/ymax)*ph
            bx = gx + ci*(bar_w+bar_gap)
            by = base - bh
            highlight = (ci == n_bars-1)
            stroke = INK if highlight else "none"
            sw = 1.6 if highlight else 0
            s.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
                     f'rx="3" fill="{col}" stroke="{stroke}" stroke-width="{sw}"/>')
            s.append(f'<text x="{bx+bar_w/2:.1f}" y="{by-6:.1f}" font-family="{FONT}" '
                     f'font-size="11" font-weight="{700 if highlight else 500}" '
                     f'fill="{INK if highlight else SUB}" text-anchor="middle">{v:.1f}</text>')
        # 그룹 라벨
        s.append(text(gx + (group_w-40)/2, base+26, bench, 14, INK, "700", "middle"))

    # 축선
    s.append(f'<line x1="{px}" y1="{base}" x2="{px+pw}" y2="{base}" stroke="{SUB}" stroke-width="1.6"/>')

    # +15.0 격차 주석 (평균 그룹: Prop 차단 48.7 -> Prop 허용 63.7)
    gi = 4  # 평균
    gx = px + gi*group_w + 20
    x_low = gx + 2*(bar_w+bar_gap) + bar_w/2   # Inf 차단·Prop 차단 (48.7)
    x_high = gx + 4*(bar_w+bar_gap) + bar_w/2  # 기본 (63.7)
    y_low = base - (48.7/ymax)*ph
    y_high = base - (63.7/ymax)*ph
    y_ann = y_high - 46
    s.append(f'<path d="M {x_low:.1f} {y_low:.1f} L {x_low:.1f} {y_ann:.1f} L {x_high:.1f} {y_ann:.1f} L {x_high:.1f} {y_high:.1f}" '
             f'fill="none" stroke="{GREEN}" stroke-width="1.8"/>')
    s.append(f'<rect x="{(x_low+x_high)/2-92:.1f}" y="{y_ann-30:.1f}" width="184" height="26" rx="13" '
             f'fill="#ecfdf5" stroke="{GREEN}" stroke-width="1.2"/>')
    s.append(text((x_low+x_high)/2, y_ann-12,
                  "지속 wiki로 평균 +15.0점", 12.5, "#047857", "800", "middle"))

    # 범례
    ly = H-70
    lx = px
    for label, col, _ in configs:
        w_est = 11 + len(label)*7.6
        extra = f' stroke="{INK}" stroke-width="1.4"' if label.endswith("(기본)") else ''
        s.append(f'<rect x="{lx}" y="{ly-13}" width="16" height="16" rx="4" fill="{col}"{extra}/>')
        s.append(text(lx+22, ly, label, 12.5, SUB, "600"))
        lx += 24 + w_est
    s.append(watermark(W, H))
    s.append('</svg>')
    (OUT / "wikiskill-ablation.svg").write_text("".join(s), encoding="utf-8")


# ══════════════════════════════════════════════════════════════
# COVER : 저장소 3개 개념도
# ══════════════════════════════════════════════════════════════
def build_cover():
    W, H = 1200, 630
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    s.append(defs())
    s.append(f'<rect width="{W}" height="{H}" fill="url(#gCover)"/>')
    s.append(f'<rect width="{W}" height="{H}" fill="none"/>')

    # 상단 태그
    s.append(card(60, 56, 190, 40, fill="#ffffff", stroke=BORDER, rx=20, sw=1.2))
    s.append(text(155, 82, "논문 리뷰 · AI 에이전트", 14, INDIGO, "700", "middle"))

    # 제목
    s.append(text(60, 168, "WikiSkill", 66, "url(#gTitle)", "800"))
    s.append(text(60, 214, "에이전트 경험을 지속 지식으로 컴파일하는 스킬 진화", 25, INK, "700"))
    s.append(text(60, 248, "실행 기록 · 누적 지식 · 실행 스킬을 서로 다른 저장소로 분리한다", 16, SUB, "500"))

    # 세 저장소 카드
    y0 = 300
    cw, ch = 340, 250
    gap = 40
    x0 = 60

    # raw 카드: 흐릿하게 쌓인 로그 더미
    s.append(card(x0, y0, cw, ch, fill="#ffffff", stroke=BORDER, rx=20, sw=1.5))
    s.append(f'<rect x="{x0}" y="{y0}" width="{cw}" height="7" rx="3.5" fill="{RAW_C}"/>')
    s.append(text(x0+28, y0+48, "raw/", 26, RAW_C, "800"))
    s.append(text(x0+28, y0+74, "원본 실행 trace", 14, SUB, "600"))
    for i in range(5):
        yy = y0+100+i*26
        op = 0.28 + i*0.14
        s.append(f'<rect x="{x0+28}" y="{yy}" width="{cw-90-i*8}" height="14" rx="4" '
                 f'fill="{RAW_C}" opacity="{op:.2f}"/>')

    # wiki 카드: 정리된 지식 페이지 묶음
    x1 = x0+cw+gap
    s.append(card(x1, y0, cw, ch, fill="#ffffff", stroke=BORDER, rx=20, sw=1.5))
    s.append(f'<rect x="{x1}" y="{y0}" width="{cw}" height="7" rx="3.5" fill="{WIKI_C}"/>')
    s.append(text(x1+28, y0+48, "wiki/", 26, WIKI_C, "800"))
    s.append(text(x1+28, y0+74, "누적되는 지식 저장소", 14, SUB, "600"))
    for i in range(3):
        yy = y0+96+i*46
        s.append(card(x1+28, yy, cw-56, 36, fill="#eef2ff", stroke="#c7d2fe", rx=8, sw=1))
        s.append(f'<circle cx="{x1+46}" cy="{yy+18}" r="5" fill="{WIKI_C}"/>')
        s.append(f'<rect x="{x1+60}" y="{yy+13}" width="{cw-120}" height="10" rx="5" fill="{WIKI_C}" opacity="0.4"/>')

    # skills 카드: 실행 지침서 한 장
    x2 = x1+cw+gap
    s.append(card(x2, y0, cw, ch, fill="#ffffff", stroke=BORDER, rx=20, sw=1.5))
    s.append(f'<rect x="{x2}" y="{y0}" width="{cw}" height="7" rx="3.5" fill="{SKILL_C}"/>')
    s.append(text(x2+28, y0+48, "skills/", 26, SKILL_C, "800"))
    s.append(text(x2+28, y0+74, "활성 실행 스킬", 14, SUB, "600"))
    s.append(card(x2+90, y0+96, cw-180, 130, fill="#ecfdf5", stroke="#a7f3d0", rx=10, sw=1.2))
    for i in range(5):
        yy = y0+116+i*22
        w = (cw-180-40) if i not in (0,) else (cw-180-90)
        s.append(f'<rect x="{x2+108}" y="{yy}" width="{w}" height="8" rx="4" '
                 f'fill="{SKILL_C}" opacity="{0.75 if i==0 else 0.35}"/>')

    # 카드 사이 화살표 (실선: 앞으로, 점선: skills만 롤백)
    ay = y0+ch/2
    s.append(f'<line x1="{x0+cw+4}" y1="{ay-10}" x2="{x1-4}" y2="{ay-10}" stroke="{SUB}" stroke-width="2.4" marker-end="url(#arrow)"/>')
    s.append(f'<line x1="{x1+cw+4}" y1="{ay-10}" x2="{x2-4}" y2="{ay-10}" stroke="{SKILL_C}" stroke-width="2.4" marker-end="url(#arrowG)"/>')
    # skills -> (되돌림) 점선
    s.append(f'<line x1="{x2-4}" y1="{ay+18}" x2="{x1+cw+4}" y2="{ay+18}" stroke="{ROSE}" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#arrowR)"/>')

    s.append(watermark(W, H))
    s.append('</svg>')
    (OUT / "wikiskill-cover.svg").write_text("".join(s), encoding="utf-8")


if __name__ == "__main__":
    build_architecture()
    build_ablation()
    build_cover()
    print("SVG 3종 생성 완료:", OUT)
