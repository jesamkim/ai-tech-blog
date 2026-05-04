# Blog Diagram Style Guide

블로그 포스트에 삽입하는 PNG 다이어그램을 일관되게 만들기 위한 테마 시스템입니다. `scripts/blog_diagram_theme.py` 한 모듈을 import하면 3종 테마(`minimal`, `vibrant`, `editorial`)를 골라 쓸 수 있습니다.

---

## 테마 선택 가이드

| 테마 | 분위기 | 권장 용도 |
|------|--------|----------|
| **minimal** | 순백 배경 + 얇은 테두리 + 단일 accent | 개념도, 의사결정 트리, 전/후 비교, 교과서 느낌의 해설 그림 |
| **vibrant** | 딥 네이비 배경 + 카테고리별 비비드 색 + 드롭섀도우 | 시스템 아키텍처, 플로우, 계층 구조, 기존 블로그 톤과의 연속성 |
| **editorial** | 크림 배경 + 직각 테두리 + 세리프 타이포 | 데이터/통계, 트렌드 분석, 정제된 인포그래픽 |

- 포스트 1개 안에서는 **한 테마로 통일** — 혼용하면 조잡해짐
- 같은 시리즈 여러 편은 같은 테마를 유지하면 브랜딩 효과
- 논문 리뷰·벤치마크는 `editorial`, 아키텍처 deep dive는 `vibrant`, API/개념 설명은 `minimal`이 잘 맞음

---

## 설치 / 전제조건

- **matplotlib 3.x** (repo에는 3.10 확인됨)
- **한국어 폰트** — 우분투 기준:
  ```bash
  sudo apt install fonts-nanum
  fc-cache -fv
  ```
  필요한 파일: `NanumSquareRound`(minimal/vibrant), `NanumMyeongjo`(editorial), `NanumGothicCoding`(mono).
- 폰트가 없으면 `DejaVu Sans`로 자동 fallback하고 경고를 1회 출력합니다.
- 추가 pip 의존성은 없습니다 — pure matplotlib만 사용.

---

## Hello World

```python
from blog_diagram_theme import setup_figure, draw_box, draw_arrow, add_title

fig, ax = setup_figure('vibrant', figsize=(12, 6))
add_title(ax, '서비스 흐름', 'Client → Server', theme='vibrant')
draw_box(ax, 1, 2, 4, 2, theme='vibrant', variant='primary',
         title='Client', subtitle='웹·모바일')
draw_box(ax, 7, 2, 4, 2, theme='vibrant', variant='accent',
         title='Server', subtitle='Agent 런타임')
draw_arrow(ax, (5, 3), (7, 3), theme='vibrant', style='thick')

fig.savefig('hello.png', dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
```

`xlim`/`ylim`은 기본 `(0,14) × (0,8)` — 기존 블로그 다이어그램과 동일한 좌표계.

---

## 3종 테마 미리보기

각 테마에 대해 동일한 3종 시나리오(아키텍처 / 의사결정 트리 / 플로우차트)를 생성했습니다. 블로그 width(약 1400px)에서 어떻게 보이는지 그대로 확인할 수 있습니다.

### Minimal

![minimal architecture](/ai-tech-blog/images/theme-gallery/minimal-architecture.png)
![minimal decision tree](/ai-tech-blog/images/theme-gallery/minimal-decision-tree.png)
![minimal flowchart](/ai-tech-blog/images/theme-gallery/minimal-flowchart.png)

### Vibrant

![vibrant architecture](/ai-tech-blog/images/theme-gallery/vibrant-architecture.png)
![vibrant decision tree](/ai-tech-blog/images/theme-gallery/vibrant-decision-tree.png)
![vibrant flowchart](/ai-tech-blog/images/theme-gallery/vibrant-flowchart.png)

### Editorial

![editorial architecture](/ai-tech-blog/images/theme-gallery/editorial-architecture.png)
![editorial decision tree](/ai-tech-blog/images/theme-gallery/editorial-decision-tree.png)
![editorial flowchart](/ai-tech-blog/images/theme-gallery/editorial-flowchart.png)

갤러리를 다시 만들려면 `python3 scripts/diagram_theme_gallery.py`.

---

## API 레퍼런스

### `setup_figure(theme, figsize, xlim, ylim, watermark)`

Figure / Axes를 만들고 배경·폰트·축을 테마에 맞춰 세팅. `watermark=''`로 워터마크 끌 수 있음.

```python
fig, ax = setup_figure('minimal', figsize=(14, 8),
                       xlim=(0, 14), ylim=(0, 8),
                       watermark='jesamkim.github.io')
```

### `draw_box(ax, x, y, w, h, *, theme, variant, title, subtitle, body, title_align)`

라운드(또는 editorial 테마에서는 직각) 박스를 그림. `variant`로 색상 카테고리 선택:

| variant | minimal | vibrant | editorial |
|--------|---------|---------|-----------|
| `primary` | 틸 테두리 | cyan 테두리 | 딥 네이비 |
| `secondary` | 슬레이트 | purple | 웜 그레이 |
| `accent` | 스카이블루 | green | 머스타드 |
| `muted` | 라이트 gray | slate blue | stone |
| `danger` | 다크 레드 | red | 버건디 |
| `success` | 다크 그린 | amber | 다크 그린 |

```python
draw_box(ax, 1, 4, 4, 1.4, theme='vibrant', variant='primary',
         title='API Gateway', subtitle='HTTP/v2',
         body=['Cognito + JWT 인증', 'Rate Limit'],
         title_align='left')
```

- `body`를 쓰면 `title_align='left'`를 권장 (본문 정렬과 일관성)
- 3줄 이상은 박스 높이를 `>= 1.8` 확보하거나 한 줄로 합치는 편이 깔끔

### `draw_arrow(ax, start, end, *, theme, style, color, label, label_offset)`

화살표. `style` 5종:

- `straight` — 기본 직선
- `curved` — 곡선(`arc3,rad=0.22`)
- `dashed` — 점선
- `thick` — 굵은 선 + 큰 화살촉
- `branching` — 직각 분기(`angle3`)

```python
draw_arrow(ax, (5, 4.7), (8, 4.7), theme='vibrant', style='thick',
           label='request', label_offset=(0, 0.3))
```

라벨은 canvas 색으로 backdrop된 둥근 박스에 그려져 화살표와 겹쳐도 읽힘.

### `draw_connector(ax, start, end, *, theme, kind)`

화살촉 없는 연결선. `kind`:

- `l-shape` — 수평 → 수직
- `step` — 중간 지점에서 피벗

```python
draw_connector(ax, (2, 5), (10, 2), theme='editorial', kind='step')
```

### `add_title(ax, text, subtitle, *, theme, y)`

Axes 상단 가운데에 제목 + 서브타이틀.

### `add_annotation(ax, x, y, text, *, theme, position, color, weight, background)`

인라인 설명. `background=True`면 화살표·선 위에 겹쳐도 읽을 수 있게 bg 처리.

### `apply_theme(ax, theme)`

기존 `ax`에 테마 배경·폰트 재적용. 외부 코드로 만든 fig에 테마만 얹고 싶을 때.

### `THEMES` (dict)

각 테마의 전체 스펙을 담은 dict. 팔레트를 직접 참조해서 matplotlib 기본 함수(`ax.plot`, `ax.scatter` 등)와 함께 쓸 때 유용:

```python
from blog_diagram_theme import THEMES
line_color = THEMES['vibrant']['palette']['primary']['line']
ax.plot(x, y, color=line_color, linewidth=2)
```

### `ThemeNotFound`

알 수 없는 테마 이름을 주면 발생. 에러 메시지에 가능한 이름 목록이 포함됩니다.

---

## 마이그레이션 팁

기존 `diagram_agentcore_*.py` 계열 스크립트를 테마 시스템으로 옮길 때:

1. **상수 블록 제거** — `BG = '#1a1a2e'`, `BLUE = '#4fc3f7'` 등 하드코딩 컬러를 모두 삭제.
2. **figure/ax 생성 치환**:
   ```python
   # before
   plt.rcParams['font.family'] = 'NanumSquareRound'
   fig, ax = plt.subplots(figsize=(14, 8))
   fig.patch.set_facecolor(BG)
   ax.set_facecolor(BG)
   ax.set_xlim(0, 14); ax.set_ylim(0, 8); ax.axis('off')

   # after
   fig, ax = setup_figure('vibrant', figsize=(14, 8))
   ```
3. **FancyBboxPatch → draw_box**:
   ```python
   # before
   box = FancyBboxPatch((x, y - h), col_w - 0.2, h,
                        boxstyle='round,pad=0.02,rounding_size=0.08',
                        linewidth=1.2, edgecolor=BLUE, facecolor=PANEL)
   ax.add_patch(box)
   ax.text(x + 0.35, y - 0.3, '타이틀', color=TEXT, fontweight='bold')

   # after
   draw_box(ax, x, y - h, col_w - 0.2, h,
            theme='vibrant', variant='primary',
            title='타이틀', title_align='left')
   ```
4. **annotate → draw_arrow**:
   ```python
   # before
   ax.annotate('', xy=end, xytext=start,
               arrowprops=dict(arrowstyle='->', color=BLUE, lw=2))

   # after
   draw_arrow(ax, start, end, theme='vibrant', style='thick')
   ```
5. **워터마크 제거** — `setup_figure`가 자동으로 달아주므로 기존 `fig.text(...)` 블록 삭제.
6. **색상 매핑** — 기존 `BLUE/GREEN/PURPLE/RED/YELLOW/ORANGE`는 대부분 `primary / accent / secondary / danger / success`로 1:1 대응됨. 매핑이 애매하면 `THEMES['vibrant']['palette']`를 직접 열어보고 가장 가까운 variant를 고르면 됨.

마이그레이션이 끝났으면 원래 상수 블록(`BG`, `PANEL`, `TEXT` 등)이 전부 사라져야 합니다 — 남아 있으면 아직 테마화가 덜 된 것.

---

## 트러블슈팅

**한국어가 네모(tofu)로 보임**
- `fc-list | grep -i nanum` 결과가 비어 있으면 폰트 미설치. `sudo apt install fonts-nanum` 후 `fc-cache -fv`.
- matplotlib 폰트 캐시가 오래됐으면 `rm -rf ~/.cache/matplotlib/` 후 재실행.

**editorial 테마에서 본문이 세리프로 안 나옴**
- `NanumMyeongjo`가 미설치일 때 `DejaVu Serif`로 대체되며 경고가 찍힘.
- Editorial 느낌을 포기하거나, 다른 세리프 폰트를 `THEMES['editorial']['fonts']['family']` 리스트 앞에 추가.

**body 텍스트가 박스 밖으로 나감**
- `draw_box`의 body 라인 간격은 data unit 기준 ~0.22. 3줄이면 박스 높이가 최소 `1.2` 필요.
- 라인 수를 줄이거나, 박스 높이를 올리거나, `·`로 합쳐 한 줄로 정리.

**화살표가 박스 경계에 정확히 안 닿음 / 너무 파고듦**
- `FancyArrowPatch`는 `shrinkA/shrinkB=2`만 적용. 박스 쪽으로 몇 픽셀 여백이 필요하면 `start`/`end` 좌표를 0.02~0.05 정도 밖으로 밀어주면 됨.

**폰트 경고 중복 출력**
- 같은 families 조합은 한 번만 경고. 다른 families로 호출하면 다시 한 번 경고.

**갤러리 재생성**
```bash
cd /Workshop/yan/ai-tech-blog
python3 scripts/blog_diagram_theme.py    # self-test
python3 scripts/diagram_theme_gallery.py # 9 PNG 생성
```
