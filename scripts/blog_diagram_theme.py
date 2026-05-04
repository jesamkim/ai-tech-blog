"""Blog diagram theme system.

Three presets (``minimal``, ``vibrant``, ``editorial``) plus shared helpers for
drawing boxes, arrows, connectors, titles, and annotations on matplotlib axes.

Typical usage::

    from blog_diagram_theme import setup_figure, draw_box, draw_arrow, add_title

    fig, ax = setup_figure('vibrant', figsize=(14, 8))
    add_title(ax, '에이전트 개발 스택 비교', 'Traditional vs Managed')
    draw_box(ax, 1, 4, 4, 1.4, theme='vibrant', variant='primary',
             title='API Gateway', subtitle='REST + WebSocket')
    draw_box(ax, 8, 4, 4, 1.4, theme='vibrant', variant='accent',
             title='Agent Service', subtitle='Strands runtime')
    draw_arrow(ax, (5.0, 4.7), (8.0, 4.7), theme='vibrant', style='thick')
    fig.savefig('out.png', dpi=200, facecolor=fig.get_facecolor())

Each theme fully owns its palette, typography, box styling, and arrow style.
Consumers pass the theme name to helpers; defaults are applied from ``THEMES``.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path


# ---------------------------------------------------------------------------
# Theme specifications
# ---------------------------------------------------------------------------

THEMES: Dict[str, Dict[str, Any]] = {
    'minimal': {
        'name': 'minimal',
        'description': 'Anthropic/Stripe docs style. White canvas, single accent, heavy whitespace.',
        'background': '#ffffff',
        'canvas': '#ffffff',
        'surface': '#ffffff',
        'grid': '#e7e5e4',
        'fonts': {
            'family': ['NanumSquareRound', 'DejaVu Sans'],
            'family_mono': ['NanumGothicCoding', 'DejaVu Sans Mono'],
            'title_weight': 'bold',
            'body_weight': 'regular',
        },
        'sizes': {'title': 18, 'subtitle': 11, 'box_title': 11, 'box_subtitle': 9, 'body': 9, 'label': 8.5},
        'text': {
            'primary': '#111827',
            'secondary': '#52525b',
            'muted': '#a1a1aa',
            'inverse': '#fafaf9',
        },
        'palette': {
            'primary':   {'line': '#0f766e', 'fill': '#ffffff', 'text': '#0f766e'},   # teal
            'secondary': {'line': '#52525b', 'fill': '#ffffff', 'text': '#27272a'},   # slate
            'accent':    {'line': '#0ea5e9', 'fill': '#ffffff', 'text': '#0369a1'},   # sky
            'muted':     {'line': '#d4d4d8', 'fill': '#fafafa', 'text': '#52525b'},   # zinc light
            'danger':    {'line': '#b91c1c', 'fill': '#ffffff', 'text': '#991b1b'},   # red subdued
            'success':   {'line': '#166534', 'fill': '#ffffff', 'text': '#166534'},   # green subdued
        },
        'box': {
            'linewidth': 1.1,
            'boxstyle': 'round,pad=0.02,rounding_size=0.05',
            'shadow': False,
        },
        'arrow': {
            'color': '#3f3f46',
            'linewidth': 1.1,
            'head_length': 8,
            'head_width': 6,
        },
    },
    'vibrant': {
        'name': 'vibrant',
        'description': 'Current blog style refined. Deep navy canvas, bold color-coded boxes.',
        'background': '#1a1a2e',
        'canvas': '#1a1a2e',
        'surface': '#16213e',
        'grid': '#30365f',
        'fonts': {
            'family': ['NanumSquareRound', 'DejaVu Sans'],
            'family_mono': ['NanumGothicCoding', 'DejaVu Sans Mono'],
            'title_weight': 'bold',
            'body_weight': 'regular',
        },
        'sizes': {'title': 18, 'subtitle': 11, 'box_title': 11, 'box_subtitle': 9, 'body': 9.5, 'label': 9},
        'text': {
            'primary': '#e0e0e0',
            'secondary': '#9aa5ce',
            'muted': '#6c7086',
            'inverse': '#1a1a2e',
        },
        'palette': {
            'primary':   {'line': '#4fc3f7', 'fill': '#16213e', 'text': '#e0e0e0'},   # blue
            'secondary': {'line': '#ab47bc', 'fill': '#1c0e28', 'text': '#e0e0e0'},   # purple
            'accent':    {'line': '#66bb6a', 'fill': '#0d2818', 'text': '#e0e0e0'},   # green
            'muted':     {'line': '#30365f', 'fill': '#16213e', 'text': '#9aa5ce'},   # slate blue
            'danger':    {'line': '#ef5350', 'fill': '#2a0e13', 'text': '#e0e0e0'},   # red
            'success':   {'line': '#ffd54f', 'fill': '#2a2410', 'text': '#e0e0e0'},   # amber
        },
        'box': {
            'linewidth': 1.5,
            'boxstyle': 'round,pad=0.02,rounding_size=0.08',
            'shadow': True,
            'shadow_color': '#0b0b18',
        },
        'arrow': {
            'color': '#9aa5ce',
            'linewidth': 1.8,
            'head_length': 10,
            'head_width': 8,
        },
    },
    'editorial': {
        'name': 'editorial',
        'description': 'NYT/Bloomberg inspired. Cream canvas, rectilinear boxes, muted deep colors.',
        'background': '#fef9f2',
        'canvas': '#fef9f2',
        'surface': '#fbf3e4',
        'grid': '#d6d3d1',
        'fonts': {
            'family': ['NanumMyeongjo', 'DejaVu Serif'],
            'family_mono': ['NanumGothicCoding', 'DejaVu Sans Mono'],
            'title_weight': 'bold',
            'body_weight': 'regular',
        },
        'sizes': {'title': 19, 'subtitle': 11, 'box_title': 11, 'box_subtitle': 9, 'body': 9.5, 'label': 8.5},
        'text': {
            'primary': '#1c1917',
            'secondary': '#57534e',
            'muted': '#a8a29e',
            'inverse': '#fef9f2',
        },
        'palette': {
            'primary':   {'line': '#1e3a8a', 'fill': '#fef9f2', 'text': '#1e3a8a'},   # deep navy
            'secondary': {'line': '#57534e', 'fill': '#fef9f2', 'text': '#292524'},   # warm gray
            'accent':    {'line': '#ca8a04', 'fill': '#fef9f2', 'text': '#854d0e'},   # mustard
            'muted':     {'line': '#d6d3d1', 'fill': '#faf5ec', 'text': '#57534e'},   # stone
            'danger':    {'line': '#991b1b', 'fill': '#fef9f2', 'text': '#991b1b'},   # burgundy
            'success':   {'line': '#14532d', 'fill': '#fef9f2', 'text': '#14532d'},   # dark green
        },
        'box': {
            'linewidth': 1.4,
            'boxstyle': 'square,pad=0.02',
            'shadow': False,
        },
        'arrow': {
            'color': '#1c1917',
            'linewidth': 1.1,
            'head_length': 7,
            'head_width': 5,
        },
    },
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ThemeNotFound(KeyError):
    """Raised when a theme name is not registered."""


def _get_theme(theme: str) -> Dict[str, Any]:
    if theme not in THEMES:
        available = ', '.join(sorted(THEMES.keys()))
        raise ThemeNotFound(f"'{theme}' — 사용 가능: {available}")
    return THEMES[theme]


# ---------------------------------------------------------------------------
# Font resolution with fallback
# ---------------------------------------------------------------------------

_FONT_WARNED: set = set()


def _pick_font(families: Iterable[str]) -> str:
    """Return the first installed font family; warn if none match."""
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in families:
        if name in installed:
            return name
    fallback = 'DejaVu Sans'
    key = tuple(families)
    if key not in _FONT_WARNED:
        warnings.warn(
            f"blog_diagram_theme: 요청한 폰트({', '.join(families)}) 미발견 — {fallback}로 대체",
            stacklevel=3,
        )
        _FONT_WARNED.add(key)
    return fallback


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def setup_figure(
    theme: str = 'vibrant',
    figsize: Tuple[float, float] = (14, 8),
    xlim: Tuple[float, float] = (0, 14),
    ylim: Tuple[float, float] = (0, 8),
    watermark: str = 'jesamkim.github.io',
):
    """Create ``(fig, ax)`` with background, fonts, and limits pre-applied.

    Parameters
    ----------
    theme : str
        One of ``minimal``, ``vibrant``, ``editorial``.
    figsize : tuple
        Passed to ``plt.subplots``; aim for 14x8-ish for blog width.
    xlim, ylim : tuple
        Axis ranges; defaults match the existing blog scripts.
    watermark : str
        Bottom-right watermark. Pass ``''`` to disable.
    """
    spec = _get_theme(theme)
    family = _pick_font(spec['fonts']['family'])

    plt.rcParams['font.family'] = family
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(spec['canvas'])
    ax.set_facecolor(spec['canvas'])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis('off')

    if watermark:
        fig.text(
            0.99, 0.012, watermark,
            ha='right', va='bottom',
            color=spec['text']['muted'],
            fontsize=8, alpha=0.7, fontstyle='italic',
        )

    return fig, ax


def apply_theme(ax, theme: str) -> None:
    """Re-apply theme background/font to an existing axes.

    Useful when reusing an axes created by other code.
    """
    spec = _get_theme(theme)
    family = _pick_font(spec['fonts']['family'])
    plt.rcParams['font.family'] = family
    ax.set_facecolor(spec['canvas'])
    fig = ax.get_figure()
    if fig is not None:
        fig.patch.set_facecolor(spec['canvas'])


def _resolve_variant(theme_spec: Dict[str, Any], variant: str) -> Dict[str, str]:
    palette = theme_spec['palette']
    if variant not in palette:
        available = ', '.join(palette.keys())
        raise ValueError(f"variant '{variant}' — 사용 가능: {available}")
    return palette[variant]


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    theme: str = 'vibrant',
    variant: str = 'primary',
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    body: Optional[Iterable[str]] = None,
    title_align: str = 'center',
    return_patch: bool = False,
):
    """Draw a themed box with optional title/subtitle/body lines.

    Parameters
    ----------
    x, y : float
        Lower-left corner in data coordinates.
    w, h : float
        Width and height.
    variant : str
        One of ``primary``, ``secondary``, ``accent``, ``muted``, ``danger``, ``success``.
    title : str or None
        Bold title rendered near the top center (or left if ``title_align='left'``).
    subtitle : str or None
        Lighter subtitle one line below the title.
    body : iterable of str or None
        Additional lines rendered beneath the subtitle, left-aligned.
    title_align : str
        ``'center'`` (default) or ``'left'``.
    return_patch : bool
        If True, return the created patch for further styling.
    """
    spec = _get_theme(theme)
    colors = _resolve_variant(spec, variant)

    box_kwargs = dict(
        boxstyle=spec['box']['boxstyle'],
        linewidth=spec['box']['linewidth'],
        edgecolor=colors['line'],
        facecolor=colors['fill'],
    )

    # Optional shadow for vibrant
    if spec['box'].get('shadow'):
        shadow = FancyBboxPatch(
            (x + 0.04, y - 0.05), w, h,
            boxstyle=spec['box']['boxstyle'],
            linewidth=0, edgecolor='none',
            facecolor=spec['box'].get('shadow_color', '#000000'),
            alpha=0.35, zorder=1,
        )
        ax.add_patch(shadow)

    box = FancyBboxPatch((x, y), w, h, zorder=2, **box_kwargs)
    ax.add_patch(box)

    sizes = spec['sizes']
    text_color = colors['text']
    sub_color = spec['text']['secondary']

    # Title
    body_items = list(body) if body else []
    has_body = bool(body_items)
    has_subtitle = bool(subtitle)

    # Vertical layout: pack from top. Data-unit offsets calibrated for
    # ylim spans around 8-9 units (blog-standard figures).
    top_y = y + h - 0.18
    if title:
        tx = x + w / 2 if title_align == 'center' else x + 0.25
        ax.text(
            tx, top_y, title,
            ha=title_align, va='top',
            color=text_color, fontsize=sizes['box_title'],
            fontweight=spec['fonts']['title_weight'], zorder=3,
        )
        cursor = top_y - 0.26
    else:
        cursor = top_y

    if subtitle:
        sx = x + w / 2 if title_align == 'center' else x + 0.25
        ax.text(
            sx, cursor, subtitle,
            ha=title_align, va='top',
            color=sub_color, fontsize=sizes['box_subtitle'],
            fontstyle='italic', zorder=3,
        )
        cursor -= 0.26

    if has_body:
        for line in body_items:
            ax.text(
                x + 0.25, cursor, line,
                ha='left', va='top',
                color=spec['text']['primary'],
                fontsize=sizes['body'], zorder=3,
            )
            cursor -= 0.22

    if return_patch:
        return box
    return None


def _arrow_kwargs(spec: Dict[str, Any], style: str, color: Optional[str]) -> Dict[str, Any]:
    arrow_spec = spec['arrow']
    use_color = color or arrow_spec['color']
    lw = arrow_spec['linewidth']
    hl = arrow_spec['head_length']
    hw = arrow_spec['head_width']

    style_map = {
        'straight':  dict(arrowstyle=f'-|>,head_length={hl},head_width={hw}', lw=lw,
                          color=use_color, shrinkA=2, shrinkB=2,
                          connectionstyle='arc3,rad=0'),
        'curved':    dict(arrowstyle=f'-|>,head_length={hl},head_width={hw}', lw=lw,
                          color=use_color, shrinkA=2, shrinkB=2,
                          connectionstyle='arc3,rad=0.22'),
        'dashed':    dict(arrowstyle=f'-|>,head_length={hl},head_width={hw}', lw=lw,
                          color=use_color, shrinkA=2, shrinkB=2,
                          connectionstyle='arc3,rad=0', linestyle='--'),
        'thick':     dict(arrowstyle=f'-|>,head_length={hl + 2},head_width={hw + 2}',
                          lw=lw + 1.2, color=use_color, shrinkA=2, shrinkB=2,
                          connectionstyle='arc3,rad=0'),
        'branching': dict(arrowstyle=f'-|>,head_length={hl},head_width={hw}', lw=lw,
                          color=use_color, shrinkA=2, shrinkB=2,
                          connectionstyle='angle3,angleA=0,angleB=90'),
    }
    if style not in style_map:
        raise ValueError(f"style '{style}' — 사용 가능: {', '.join(style_map.keys())}")
    return style_map[style]


def draw_arrow(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    theme: str = 'vibrant',
    style: str = 'straight',
    color: Optional[str] = None,
    label: Optional[str] = None,
    label_offset: Tuple[float, float] = (0.15, 0.15),
):
    """Draw a themed arrow between two points.

    ``style``: ``straight``, ``curved``, ``dashed``, ``thick``, ``branching``.
    """
    spec = _get_theme(theme)
    kwargs = _arrow_kwargs(spec, style, color)
    arrow = FancyArrowPatch(start, end, zorder=4, **kwargs)
    ax.add_patch(arrow)

    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mx, my, label,
            ha='left', va='center',
            color=kwargs['color'],
            fontsize=spec['sizes']['label'],
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=spec['canvas'], edgecolor='none'),
            zorder=5,
        )
    return arrow


def draw_connector(
    ax,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    theme: str = 'vibrant',
    kind: str = 'l-shape',
    color: Optional[str] = None,
):
    """Draw an L-shaped or stepped connector line (no arrow head).

    ``kind``: ``l-shape`` (horizontal then vertical) or ``step`` (midpoint pivot).
    """
    spec = _get_theme(theme)
    use_color = color or spec['arrow']['color']
    lw = spec['arrow']['linewidth']

    sx, sy = start
    ex, ey = end
    if kind == 'l-shape':
        verts = [(sx, sy), (ex, sy), (ex, ey)]
    elif kind == 'step':
        mx = (sx + ex) / 2
        verts = [(sx, sy), (mx, sy), (mx, ey), (ex, ey)]
    else:
        raise ValueError(f"kind '{kind}' — 사용 가능: l-shape, step")

    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
    patch = PathPatch(Path(verts, codes), fill=False, edgecolor=use_color,
                      linewidth=lw, zorder=3)
    ax.add_patch(patch)
    return patch


def add_title(
    ax,
    text: str,
    subtitle: Optional[str] = None,
    *,
    theme: str = 'vibrant',
    y: Optional[float] = None,
):
    """Add a centered title (+ optional subtitle) near the top of the axes."""
    spec = _get_theme(theme)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx = (xlim[0] + xlim[1]) / 2
    top = ylim[1] if y is None else y
    title_y = top - (ylim[1] - ylim[0]) * 0.04
    sub_y = title_y - (ylim[1] - ylim[0]) * 0.05

    ax.text(
        cx, title_y, text,
        ha='center', va='center',
        color=spec['text']['primary'],
        fontsize=spec['sizes']['title'],
        fontweight=spec['fonts']['title_weight'],
    )
    if subtitle:
        ax.text(
            cx, sub_y, subtitle,
            ha='center', va='center',
            color=spec['text']['secondary'],
            fontsize=spec['sizes']['subtitle'],
            fontstyle='italic',
        )


def add_annotation(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    theme: str = 'vibrant',
    position: str = 'center',
    color: Optional[str] = None,
    weight: str = 'regular',
    background: bool = False,
):
    """Add inline annotation text.

    ``position``: ``center``, ``left``, ``right``.
    Set ``background=True`` to render a canvas-colored rounded backdrop so the
    text reads cleanly over arrows/lines.
    """
    spec = _get_theme(theme)
    ha = {'center': 'center', 'left': 'left', 'right': 'right'}[position]
    use_color = color or spec['text']['secondary']

    bbox = None
    if background:
        bbox = dict(boxstyle='round,pad=0.25', facecolor=spec['canvas'], edgecolor='none')

    ax.text(
        x, y, text,
        ha=ha, va='center',
        color=use_color,
        fontsize=spec['sizes']['label'],
        fontweight=weight,
        bbox=bbox,
        zorder=5,
    )


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    errs = []
    for name in THEMES:
        try:
            fig, ax = setup_figure(name, figsize=(6, 3), xlim=(0, 10), ylim=(0, 6))
            draw_box(ax, 1, 2, 3, 2, theme=name, variant='primary',
                     title='Box', subtitle='self-test')
            draw_box(ax, 6, 2, 3, 2, theme=name, variant='accent',
                     title='Box2', subtitle='ok')
            draw_arrow(ax, (4, 3), (6, 3), theme=name, style='straight')
            add_title(ax, f'theme={name}', '자체검증')
            plt.close(fig)
            print(f'[ok] theme={name}')
        except Exception as e:  # noqa: BLE001
            errs.append((name, e))
            print(f'[FAIL] theme={name}: {e}')
    try:
        setup_figure('nonexistent')
    except ThemeNotFound as e:
        print(f'[ok] ThemeNotFound raised as expected: {e}')
    else:
        errs.append(('error-handling', 'ThemeNotFound not raised'))

    if errs:
        sys.exit(1)
    print('blog_diagram_theme self-test passed')
