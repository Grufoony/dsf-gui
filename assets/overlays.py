"""
overlays.py
───────────
The two panels burned into every recorded video frame: the colour legend and a
square traffic-load chart mirroring the one in the side panel.

Both are rendered once, when a recording starts, into QImages with alpha. Per
frame only a drawImage and the moving chart marker are paid for - about 2 ms,
against ~85 ms if the chart were re-rendered each time.

The panels float inside the frame rather than extending it, so the recorded
resolution stays exactly what the map widget grabs.

Nothing here imports MainWindow: the host passes in what it is showing, which
keeps this module renderable (and testable) without a database or a map.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen

from .database import ramp_color

try:
    # backend_agg, not backend_qtagg: the panel is never a widget, has no
    # parent, and must render identically under QT_QPA_PLATFORM=offscreen.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Palette ───────────────────────────────────────────────────────────────────

PLATE_BG = QColor(14, 17, 22, 234)  # dark plate; opaque enough to stop map
# labels from bleeding through and competing with the chart's own text
PLATE_BORDER = QColor(255, 255, 255, 46)
TEXT_PRIMARY = QColor(240, 243, 247)
TEXT_MUTED = QColor(176, 186, 199)
CHIP_BG = QColor(10, 12, 16, 200)
MARKER = QColor(230, 57, 70)  # the same red as ChartWidget's axvline
ACCENT_HEX = "#4aa3ff"  # brighter sibling of the side chart's #3a7ebf

# ── Geometry ──────────────────────────────────────────────────────────────────

MARGIN = 14  # panel inset from the frame edge, at scale 1
CORNER_RADIUS = 9
PAD = 10

# Chart side as a fraction of frame height; the selector in the UI picks one.
CHART_SIZES: dict[str, float] = {
    "No chart": 0.0,
    "Small": 0.24,
    "Medium": 0.32,
    "Large": 0.40,
}
DEFAULT_CHART_SIZE = "Medium"
CHART_MIN, CHART_MAX = 150, 360
CHART_MAX_FRACTION = 0.45  # never eat more than this of the short edge

LEGEND_FRACTION = 0.28
LEGEND_MIN_W, LEGEND_MAX_W = 200, 340

MIN_FRAME = QSize(360, 300)  # below this, no overlays at all
CHART_DPI = 100

PRETTY_METRIC = {
    "mean_density_vpk": "Mean density (veh/km)",
    "mean_speed_kph": "Mean speed (km/h)",
    "mean_queue_length": "Mean queue length",
    "mean_travel_time_s": "Mean travel time (s)",
    "n_agents": "Vehicles in network",
    "n_ghost_agents": "Ghost agents",
    "std_speed_kph": "Speed spread (km/h)",
    "std_density_vpk": "Density spread (veh/km)",
}


def pretty_metric(col: str) -> str:
    """A human title for an avg_stats column name."""
    # The column set comes from PRAGMA table_info(avg_stats), so unknown names
    # are expected rather than exceptional.
    return PRETTY_METRIC.get(col, col.replace("_", " ").capitalize())


def format_tick(v: float) -> str:
    """Tick/readout formatting, shared with the on-screen LegendWidget."""
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _compact(v: float, _pos=None) -> str:
    a = abs(v)
    if a >= 1_000_000:
        return f"{v / 1e6:.1f}M"
    if a >= 1_000:
        return f"{v / 1e3:.1f}k"
    return format_tick(v)


def scale_for(size: QSize) -> float:
    """1.0 at a ~700 px frame; clamped so text never becomes unreadable."""
    return max(0.75, min(1.6, min(size.width(), size.height()) / 700.0))


def _draw_plate(p: QPainter, rect: QRectF, radius: float) -> None:
    p.setPen(Qt.NoPen)
    p.setBrush(PLATE_BG)
    p.drawRoundedRect(rect, radius, radius)
    p.setPen(QPen(PLATE_BORDER, 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


# ── Legend panel ──────────────────────────────────────────────────────────────


def render_legend_panel(
    *,
    width: int,
    label: str,
    domain: tuple[float, float],
    reversed_: bool,
    scale: float = 1.0,
) -> QImage:
    """
    The colour legend, as a dark translucent plate of the given width.

    Mirrors LegendWidget.paintEvent stroke for stroke - the same ramp_color,
    the same three ticks, the same handling of reversed scales - but light on
    dark, because the widget's black text would vanish against the basemap.
    Calling ramp_color rather than re-deriving the gradient is what stops the
    burned-in legend from ever disagreeing with the edge colours.
    """
    title_f = QFont("Arial", max(8, round(9 * scale)), QFont.Bold)
    tick_f = QFont("Arial", max(7, round(8 * scale)))
    tm, km = QFontMetrics(title_f), QFontMetrics(tick_f)

    pad = round(PAD * scale)
    gap = round(6 * scale)
    bar_h = round(15 * scale)
    height = pad + tm.height() + gap + bar_h + gap + km.height() + pad

    img = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setRenderHint(QPainter.TextAntialiasing)
    _draw_plate(p, QRectF(0.5, 0.5, width - 1, height - 1), CORNER_RADIUS * scale)

    p.setPen(TEXT_PRIMARY)
    p.setFont(title_f)
    p.drawText(pad, pad + tm.ascent(), label)

    bx, bw = pad, width - 2 * pad
    by = pad + tm.height() + gap
    for i in range(bw):
        t = i / max(1, bw - 1)
        if reversed_:
            t = 1.0 - t
        r, g, b = ramp_color(t)
        # fillRect per column, not drawLine: an antialiased 1px line blends
        # with its neighbours and washes the bar out (same reason as the widget).
        p.fillRect(bx + i, by, 1, bar_h, QColor(r, g, b))

    p.setPen(QPen(QColor(255, 255, 255, 70), 1.0))
    p.setBrush(Qt.NoBrush)
    p.drawRect(bx, by, bw - 1, bar_h - 1)

    dmin, dmax = domain
    p.setFont(tick_f)
    p.setPen(TEXT_MUTED)
    baseline = by + bar_h + gap + km.ascent()
    lo_s, mid_s, hi_s = (
        format_tick(dmin),
        format_tick((dmin + dmax) / 2),
        format_tick(dmax),
    )
    p.drawText(bx, baseline, lo_s)
    p.drawText(bx + (bw - km.horizontalAdvance(mid_s)) // 2, baseline, mid_s)
    p.drawText(bx + bw - km.horizontalAdvance(hi_s), baseline, hi_s)
    p.end()
    return img


# ── Chart panel ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChartPanel:
    """A pre-rendered chart plus the pixel geometry its marker needs."""

    image: QImage
    x_px: tuple[float, ...]  # panel-local x of each timestep
    y_px: tuple[float, ...]  # panel-local y of the curve (Qt: top-down)
    plot_top: float
    plot_bottom: float
    plot_left: float
    plot_right: float
    values: tuple[float, ...]


def render_chart_panel(
    *,
    side: int,
    title: str,
    values: Sequence[float],
    highlight: tuple[int, int] | None = None,
    scale: float = 1.0,
) -> ChartPanel | None:
    """
    The side chart, restyled for a dark map and squared off.

    Returns None when matplotlib is missing or there is nothing to plot, so the
    caller records without the panel rather than failing the whole export.
    """
    if not _HAS_MPL or len(values) == 0 or side < 120:
        return None

    ys = [float(v) for v in values]
    xs = list(range(len(ys)))
    inches = side / CHART_DPI

    # facecolor "none" keeps the Agg buffer transparent, so the rounded plate
    # can be drawn underneath by QPainter instead of fighting matplotlib for
    # rounded corners.
    fig = Figure(figsize=(inches, inches), dpi=CHART_DPI, facecolor="none")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor((1.0, 1.0, 1.0, 0.04))

    if highlight is not None and highlight[0] < highlight[1]:
        # The recorded window, shaded. Free: it bakes into the static panel.
        ax.axvspan(highlight[0], highlight[1], color="#ffffff", alpha=0.10, lw=0)

    ax.fill_between(xs, ys, min(ys), color=ACCENT_HEX, alpha=0.22, lw=0)
    ax.plot(xs, ys, color=ACCENT_HEX, linewidth=2.0, solid_capstyle="round")

    ax.set_title(title, color="#f0f2f5", fontsize=10 * scale, pad=6)
    ax.set_xlabel("Time step", color="#aab4c0", fontsize=8 * scale, labelpad=2)
    ax.tick_params(colors="#c8d2de", labelsize=8 * scale, length=3, width=0.8)
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        ax.spines[edge].set_color("#7e8a9a")
        ax.spines[edge].set_linewidth(0.8)
    ax.grid(True, color="#ffffff", alpha=0.12, linewidth=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(FuncFormatter(_compact))
    ax.set_xlim(0, max(1, len(xs) - 1))  # same guard as ChartWidget._redraw
    fig.tight_layout(pad=1.1)
    canvas.draw()

    w, h = canvas.get_width_height()
    # transData is the authority: it already folds in tight_layout's margins,
    # xlim and the autoscaled ylim, so the marker lands on the drawn curve by
    # construction instead of via arithmetic that silently skews when the
    # styling changes. Matplotlib's y origin is the bottom, Qt's is the top.
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    x_px = tuple(float(v) for v in pts[:, 0])
    y_px = tuple(float(h - v) for v in pts[:, 1])
    bb = ax.get_window_extent()

    buf = memoryview(canvas.buffer_rgba()).tobytes()
    plot = QImage(buf, w, h, QImage.Format_RGBA8888).copy()  # copy: Qt must own it

    img = QImage(side, side, QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    _draw_plate(p, QRectF(0.5, 0.5, side - 1, side - 1), CORNER_RADIUS * scale)
    p.drawImage(0, 0, plot)
    p.end()

    return ChartPanel(
        image=img,
        x_px=x_px,
        y_px=y_px,
        plot_top=float(h - bb.y1),
        plot_bottom=float(h - bb.y0),
        plot_left=float(bb.x0),
        plot_right=float(bb.x1),
        values=tuple(ys),
    )


# ── Compositor ────────────────────────────────────────────────────────────────


class OverlayCompositor:
    """
    Both panels, rendered once and then stamped onto every frame.

    Build it with what the UI is currently showing, call prepare() with the
    frame size the recording pins, then composite() per frame. Everything that
    costs real time happens in prepare().
    """

    def __init__(
        self,
        *,
        legend_label: str,
        legend_domain: tuple[float, float],
        legend_reversed: bool,
        chart_title: str = "",
        chart_values: Sequence[float] = (),
        chart_fraction: float = CHART_SIZES[DEFAULT_CHART_SIZE],
        highlight: tuple[int, int] | None = None,
    ):
        self._legend_label = legend_label
        self._legend_domain = legend_domain
        self._legend_reversed = legend_reversed
        self._chart_title = chart_title
        self._chart_values = chart_values
        self._chart_fraction = chart_fraction
        self._highlight = highlight

        self.size: QSize | None = None
        self._scale = 1.0
        self._legend: QImage | None = None
        self._legend_origin = QPoint(0, 0)
        self._chart: ChartPanel | None = None
        self._chart_origin = QPoint(0, 0)
        self._value_font = QFont("Arial", 8)

    # ── Setup ────────────────────────────────────────────────────────────────

    def prepare(self, size: QSize) -> None:
        """Render both panels for a frame of *size* and place them."""
        self.size = size
        self._legend = None
        self._chart = None
        if size.width() < MIN_FRAME.width() or size.height() < MIN_FRAME.height():
            return

        s = self._scale = scale_for(size)
        m = round(MARGIN * s)
        self._value_font = QFont("Arial", max(7, round(8 * s)))

        legend_w = int(
            min(
                max(LEGEND_FRACTION * size.width(), LEGEND_MIN_W),
                LEGEND_MAX_W,
                size.width() - 2 * m,
            )
        )
        self._legend = render_legend_panel(
            width=legend_w,
            label=self._legend_label,
            domain=self._legend_domain,
            reversed_=self._legend_reversed,
            scale=s,
        )
        self._legend_origin = QPoint(
            size.width() - m - self._legend.width(),
            size.height() - m - self._legend.height(),
        )

        if self._chart_fraction <= 0:
            return
        side = int(
            min(
                max(self._chart_fraction * size.height(), CHART_MIN),
                CHART_MAX,
                CHART_MAX_FRACTION * min(size.width(), size.height()),
            )
        )
        # Both panels sit on the bottom edge, so they compete for width.
        # The legend is what makes the colours readable, so it keeps its space
        # and the chart is the one that gives way.
        if m + side + m > self._legend_origin.x():
            return
        chart = render_chart_panel(
            side=side,
            title=self._chart_title,
            values=self._chart_values,
            highlight=self._highlight,
            scale=s,
        )
        if chart is None:
            return
        self._chart = chart
        self._chart_origin = QPoint(m, size.height() - m - side)

    # ── Per frame ────────────────────────────────────────────────────────────

    def composite(self, frame: QImage, idx: int) -> None:
        """Stamp the panels onto *frame*, in place."""
        if self._legend is None and self._chart is None:
            return
        p = QPainter(frame)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        if self._legend is not None:
            p.drawImage(self._legend_origin, self._legend)
        if self._chart is not None:
            self._draw_chart(p, idx)
        p.end()

    def _draw_chart(self, p: QPainter, idx: int) -> None:
        c = self._chart
        assert c is not None
        ox, oy = self._chart_origin.x(), self._chart_origin.y()
        p.drawImage(self._chart_origin, c.image)

        # avg_stats and road_data are loaded by separate queries and can differ
        # in length, so idx is not guaranteed to index the series.
        i = max(0, min(len(c.x_px) - 1, idx))
        x = ox + c.x_px[i]
        y = oy + c.y_px[i]
        top, bottom = oy + c.plot_top, oy + c.plot_bottom

        p.setPen(
            QPen(
                QColor(MARKER.red(), MARKER.green(), MARKER.blue(), 235),
                max(1.5, 2.0 * self._scale),
            )
        )
        p.drawLine(QPointF(x, top), QPointF(x, bottom))

        r = 3.4 * self._scale
        p.setPen(QPen(QColor(255, 255, 255, 230), 1.4))
        p.setBrush(MARKER)
        p.drawEllipse(QPointF(x, y), r, r)

        # Value chip, flipped to the inside near the right edge so it cannot
        # fall off the panel at the last frame.
        text = _compact(c.values[i])
        fm = QFontMetrics(self._value_font)
        tw, th = fm.horizontalAdvance(text), fm.height()
        left = x + 6 if x < ox + c.image.width() / 2 else x - 6 - (tw + 10)
        left = min(max(left, ox + c.plot_left), ox + c.plot_right - tw - 10)
        chip = QRectF(left, max(top + 2, y - r - 6 - th), tw + 10, th + 4)
        p.setPen(Qt.NoPen)
        p.setBrush(CHIP_BG)
        p.drawRoundedRect(chip, 3, 3)
        p.setPen(TEXT_PRIMARY)
        p.setFont(self._value_font)
        p.drawText(chip, Qt.AlignCenter, text)
