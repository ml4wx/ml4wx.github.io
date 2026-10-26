"""Small server-side SVG charts that follow the site theme (colors come from CSS variables).

Every element also carries light-theme presentation attributes (fill, stroke, font-size) as a
fallback, so a chart still reads correctly if the stylesheet is missing or an old cached copy;
the stylesheet overrides them with the theme colors.

Specs: thin marks (bars <= 24px, 4px rounded data-end, square baseline; 2px lines; end dots r=4
with a 2px surface ring), solid hairline grid, one hue per chart, selective direct labels,
native hover titles on every mark (hit areas larger than the mark) and a table view per chart.
"""

from __future__ import annotations

import math
from html import escape


def nice_ticks(vmax: float, n: int = 4) -> list[float]:
    if vmax <= 0:
        return [0, 1]
    raw = vmax / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return [i * step for i in range(int(math.ceil(vmax / step)) + 1)]


def fmt(v: float, unit: str = "") -> str:
    if unit == "%":
        return f"{v:.0f}%"
    if v >= 1000:
        return f"{v / 1000:.1f}K".replace(".0K", "K")
    return f"{v:.0f}"


def _bar_path(x: float, y: float, w: float, base: float, r: float = 4) -> str:
    """Column with a rounded top (data end) and a square baseline."""
    h = base - y
    r = max(0.0, min(r, w / 2, h))
    return (
        f"M{x:.1f},{base:.1f}V{y + r:.1f}Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f}"
        f"H{x + w - r:.1f}Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f}V{base:.1f}Z"
    )


def table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    head = "".join(f"<th scope='col'>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f"<details class='viz-table'><summary>Table view</summary><table><caption>{escape(caption)}</caption>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></details>"
    )


def columns(data: list[tuple[str, float]], *, unit: str = "", partial_last: bool = False, label: str = "") -> str:
    """Vertical columns, one series. `partial_last`: the last column is an incomplete period."""
    w, h, left, bottom, top = 480, 190, 36, 24, 14
    base = h - bottom
    ticks = nice_ticks(max(v for _, v in data) if data else 1)
    ymax = ticks[-1]
    band = (w - left - 8) / max(1, len(data))
    bw = min(24.0, band * 0.6)
    y_of = lambda v: base - (base - top) * v / ymax  # noqa: E731
    out = [f"<svg class='viz' viewBox='0 0 {w} {h}' role='img' aria-label='{escape(label)}'>"]
    for t in ticks:
        out.append(f"<line class='viz-grid' stroke='#dde3ea' x1='{left}' x2='{w}' y1='{y_of(t):.1f}' y2='{y_of(t):.1f}'/>")
        out.append(f"<text class='viz-tick' font-size='11' fill='#5b6878' x='{left - 6}' y='{y_of(t) + 4:.1f}' text-anchor='end'>{fmt(t, unit)}</text>")
    for i, (name, v) in enumerate(data):
        x = left + band * i + (band - bw) / 2
        last = i == len(data) - 1
        partial = last and partial_last
        cls, extra = ("viz-bar partial", " opacity='0.55'") if partial else ("viz-bar", "")
        tip = f"{name}: {fmt(v, unit)}" + (" (so far)" if last and partial_last else "")
        out.append(f"<g class='viz-hit'><title>{escape(tip)}</title>"
                   f"<rect x='{left + band * i:.1f}' y='{top}' width='{band:.1f}' height='{base - top}' fill='transparent'/>"
                   f"<path class='{cls}'{extra} fill='#0b6aa2' d='{_bar_path(x, y_of(v), bw, base)}'/></g>")  # fmt: skip
        if len(data) <= 12 or i % 2 == 0 or last:
            out.append(f"<text class='viz-tick' font-size='11' fill='#5b6878' x='{x + bw / 2:.1f}' y='{h - 8}' text-anchor='middle'>{escape(name)}</text>")
        if last:
            out.append(f"<text class='viz-value' font-size='11' font-weight='600' fill='#16202c' x='{x + bw / 2:.1f}' y='{y_of(v) - 6:.1f}' text-anchor='middle'>"
                       f"{fmt(v, unit)}{'*' if partial_last else ''}</text>")  # fmt: skip
    out.append(f"<line class='viz-axis' stroke='#98a4b3' x1='{left}' x2='{w}' y1='{base}' y2='{base}'/></svg>")
    return "".join(out)


def hbars(items: list[tuple[str, float, str | None]], *, unit: str = "", label: str = "") -> str:
    """Horizontal bars, one series, value at the tip. items: (label, value, href or None)."""
    w, row, left, right = 480, 24, 170, 44
    h = row * len(items) + 8
    vmax = max((v for _, v, _ in items), default=1) or 1
    bh = 14
    out = [f"<svg class='viz' viewBox='0 0 {w} {h}' role='img' aria-label='{escape(label)}'>"]
    for i, (name, v, href) in enumerate(items):
        y = 4 + i * row
        bw = max(2.0, (w - left - right) * v / vmax)
        r = min(4, bh / 2, bw)
        path = (f"M{left},{y + (row - bh) / 2:.1f}H{left + bw - r:.1f}Q{left + bw:.1f},{y + (row - bh) / 2:.1f} "
                f"{left + bw:.1f},{y + (row - bh) / 2 + r:.1f}V{y + (row + bh) / 2 - r:.1f}Q{left + bw:.1f},"
                f"{y + (row + bh) / 2:.1f} {left + bw - r:.1f},{y + (row + bh) / 2:.1f}H{left}Z")  # fmt: skip
        text = escape(name if len(name) <= 27 else name[:26] + "…")
        name_el = f"<a href='{escape(href)}'><text class='viz-label link' font-size='12' fill='#0b6aa2' x='{left - 8}' y='{y + row / 2 + 4:.1f}' text-anchor='end'>{text}</text></a>" if href else f"<text class='viz-label' font-size='12' fill='#16202c' x='{left - 8}' y='{y + row / 2 + 4:.1f}' text-anchor='end'>{text}</text>"  # noqa: E501
        out.append(f"<g class='viz-hit'><title>{escape(name)}: {fmt(v, unit)}</title>"
                   f"<rect x='{left}' y='{y}' width='{w - left}' height='{row}' fill='transparent'/>"
                   f"<path class='viz-bar' fill='#0b6aa2' d='{path}'/></g>{name_el}"
                   f"<text class='viz-value' font-size='11' font-weight='600' fill='#16202c' x='{left + bw + 6:.1f}' y='{y + row / 2 + 4:.1f}'>{fmt(v, unit)}</text>")  # fmt: skip
    out.append(f"<line class='viz-axis' stroke='#98a4b3' x1='{left}' x2='{left}' y1='0' y2='{h}'/></svg>")
    return "".join(out)


def mini_line(series: dict[int, float], *, ymax: float, unit: str = "", label: str = "", partial_last: bool = False) -> str:
    """One small-multiple panel: a single 2px line on a shared y-scale, end dot and end value."""
    w, h, left, right, top, bottom = 220, 120, 30, 34, 10, 20
    xs = list(series)
    base = h - bottom
    x_of = lambda yr: left + (w - left - right) * (xs.index(yr) / max(1, len(xs) - 1))  # noqa: E731
    y_of = lambda v: base - (base - top) * v / ymax  # noqa: E731
    pts = " ".join(f"{x_of(yr):.1f},{y_of(v):.1f}" for yr, v in series.items())
    ticks = nice_ticks(ymax, 2)
    out = [f"<svg class='viz viz-mini' viewBox='0 0 {w} {h}' role='img' aria-label='{escape(label)}'>"]
    for t in ticks:
        if t <= ymax:
            out.append(f"<line class='viz-grid' stroke='#dde3ea' x1='{left}' x2='{w - right}' y1='{y_of(t):.1f}' y2='{y_of(t):.1f}'/>"
                       f"<text class='viz-tick' font-size='11' fill='#5b6878' x='{left - 4}' y='{y_of(t) + 3:.1f}' text-anchor='end'>{fmt(t, unit)}</text>")  # fmt: skip
    out.append(f"<polyline class='viz-line' fill='none' stroke='#0b6aa2' stroke-width='2' points='{pts}'/>")
    for yr, v in series.items():
        tip = f"{yr}: {fmt(v, unit)}" + (" (so far)" if partial_last and yr == xs[-1] else "")
        out.append(f"<g class='viz-hit'><title>{escape(tip)}</title>"
                   f"<circle cx='{x_of(yr):.1f}' cy='{y_of(v):.1f}' r='12' fill='transparent'/></g>")  # fmt: skip
    last_v = series[xs[-1]]
    out.append(f"<circle class='viz-dot' fill='#0b6aa2' cx='{x_of(xs[-1]):.1f}' cy='{y_of(last_v):.1f}' r='4'/>"
               f"<text class='viz-value' font-size='11' font-weight='600' fill='#16202c' x='{x_of(xs[-1]) + 7:.1f}' y='{y_of(last_v) + 4:.1f}'>{fmt(last_v, unit)}</text>"
               f"<text class='viz-tick' font-size='11' fill='#5b6878' x='{left}' y='{h - 5}'>{xs[0]}</text>"
               f"<text class='viz-tick' font-size='11' fill='#5b6878' x='{w - right}' y='{h - 5}' text-anchor='end'>{xs[-1]}</text>"
               f"<line class='viz-axis' stroke='#98a4b3' x1='{left}' x2='{w - right}' y1='{base}' y2='{base}'/></svg>")  # fmt: skip
    return "".join(out)


def sparkline(values: list[float], *, label: str = "") -> str:
    """Tiny trend line (de-emphasis gray) with the current value as an accent dot."""
    w, h, pad = 120, 32, 4
    vmax = max(values) or 1
    step = (w - 2 * pad) / max(1, len(values) - 1)
    pts = [(pad + i * step, h - pad - (h - 2 * pad) * v / vmax) for i, v in enumerate(values)]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    x, y = pts[-1]
    return (f"<svg class='viz-spark' viewBox='0 0 {w} {h}' role='img' aria-label='{escape(label)}'>"
            f"<polyline class='viz-spark-line' fill='none' stroke='#98a4b3' stroke-width='1.5' points='{poly}'/><circle class='viz-dot' fill='#0b6aa2' cx='{x:.1f}' cy='{y:.1f}' r='3'/></svg>")  # fmt: skip
