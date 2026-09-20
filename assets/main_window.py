"""
main_window.py
──────────────
QMainWindow that wires everything together:

  ┌─ Toolbar ──────────────────────────────────────────────────┐
  │  [Load DB]  Tiles:[▼]  Color by:[▼]  [Screenshot]          │
  ├─ QSplitter ────────────────────────────────────────┬───────┤
  │                                                    │Search │
  │               MapWidget                            │───────│
  │                                                    │Info   │
  │                                                    │───────│
  │                                                    │Legend │
  │                                                    │───────│
  │                                                    │Chart  │
  ├────────────────────────────────────────────────────┴───────┤
  │  ▶  FPS [10]  |══════════════slider══════════════|  label  │
  └────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .database import (
    DEFAULT_OBSERVABLE,
    DENSITY_NORM_SATURATION,
    EDGE_OBSERVABLE_CONFIG,
    edge_colors_for_timestep,
    get_simulations,
    load_edges,
    observable_row,
    ramp_color,
)
from .map_widget import MapWidget
from .overlays import (
    CHART_SIZES,
    DEFAULT_CHART_SIZE,
    OverlayCompositor,
    format_tick,
    pretty_metric,
)
from .sim_loader import SimulationLoader
from .video_export import FFmpegSink, VideoRecorder, even_size, ffmpeg_path, fit

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _nearest_time_index(old_datetimes: list, old_idx: int, new_datetimes: list) -> int:
    """
    The entry of *new_datetimes* closest in time-of-day to where we were.

    Switching between runs of different days should land on the same moment of
    the day, not the same row number. When both runs share a timestep grid -
    the usual case - this returns the identical index.
    """
    if not new_datetimes:
        return 0
    if not old_datetimes or not 0 <= old_idx < len(old_datetimes):
        return 0
    if len(old_datetimes) == len(new_datetimes):
        return old_idx
    target = old_datetimes[old_idx]
    seconds = target.hour * 3600 + target.minute * 60 + target.second
    return min(
        range(len(new_datetimes)),
        key=lambda i: abs(
            new_datetimes[i].hour * 3600
            + new_datetimes[i].minute * 60
            + new_datetimes[i].second
            - seconds
        ),
    )


# ── Legend widget ─────────────────────────────────────────────────────────────


class LegendWidget(QWidget):
    """Draws a blue→yellow→red gradient bar with domain labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        cfg = EDGE_OBSERVABLE_CONFIG[DEFAULT_OBSERVABLE]
        self._label = cfg["label"]
        self._domain = (0.0, DENSITY_NORM_SATURATION)  # normalised occupancy
        self._reversed = False
        self.setFixedHeight(58)
        self.setMinimumWidth(160)

    def set_observable(self, key: str, domain: tuple[float, float], reversed_: bool):
        cfg = EDGE_OBSERVABLE_CONFIG.get(key, {})
        self._label = cfg.get("label", key)
        self._domain = domain
        self._reversed = reversed_
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()

        # Title
        p.setPen(QColor(0, 0, 0))
        p.setFont(QFont("Arial", 8, QFont.Bold))
        p.drawText(2, 12, self._label)

        # Gradient bar  (y = 18..35) — same ramp the edges are coloured with
        for i in range(w):
            t = i / max(1, w - 1)
            if self._reversed:
                t = 1.0 - t
            r, g, b = ramp_color(t)
            # fillRect, not drawLine: an antialiased 1px line blends with its
            # neighbours and washes the bar out relative to the actual edges.
            p.fillRect(i, 18, 1, 17, QColor(r, g, b))

        # Border
        p.setPen(QColor(120, 120, 120))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 18, w - 1, 17)

        # Labels
        p.setFont(QFont("Arial", 7))
        p.setPen(QColor(0, 0, 0))
        dmin, dmax = self._domain
        dmid = (dmin + dmax) / 2

        # format_tick is shared with the legend burned into recorded video, so
        # the two can never disagree about how a domain is written.
        fm = p.fontMetrics()
        min_s = format_tick(dmin)
        mid_s = format_tick(dmid)
        max_s = format_tick(dmax)
        mid_x = (w - fm.horizontalAdvance(mid_s)) // 2
        max_x = w - fm.horizontalAdvance(max_s)
        p.drawText(2, 54, min_s)
        p.drawText(mid_x, 54, mid_s)
        p.drawText(max_x, 54, max_s)
        p.end()


# ── Timeline slider ───────────────────────────────────────────────────────────


class RangeSlider(QSlider):
    """
    The playback slider, with the chosen recording window shaded on its groove.

    Marking in/out points is invisible otherwise: the numbers live in a label
    off to the side, while the thing the user is actually pointing at is the
    timeline.
    """

    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._range: tuple[int, int] | None = None

    def set_marked_range(self, lo: int | None, hi: int | None):
        self._range = None if lo is None or hi is None else (lo, hi)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._range is None or self.maximum() <= self.minimum():
            return
        lo, hi = self._range

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()
        groove = style.subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )
        handle = style.subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self
        )
        # Positions are measured along the span the handle's centre can travel,
        # which is the groove inset by half a handle at each end.
        span = groove.width() - handle.width()
        if span <= 0:
            return
        origin = groove.x() + handle.width() / 2

        def pos(value: int) -> float:
            t = (value - self.minimum()) / (self.maximum() - self.minimum())
            return origin + t * span

        x0, x1 = pos(lo), pos(hi)
        band = QRectF(x0, groove.y(), max(2.0, x1 - x0), groove.height())
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(band, QColor(60, 140, 220, 110))
        p.setPen(QColor(60, 140, 220, 220))
        for x in (x0, x1):
            p.drawLine(QPointF(x, groove.y() - 2), QPointF(x, groove.bottom() + 2))
        p.end()


# ── Chart widget ──────────────────────────────────────────────────────────────


class ChartWidget(QWidget):
    """
    Time-series chart of aggregate simulation statistics.
    Requires matplotlib; shows a placeholder label when it is absent.
    Click or drag on the chart to seek to that timestep.
    """

    time_index_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._global_data: list[dict] = []
        self._current_col = "mean_density_vpk"
        self._current_idx = 0
        self._marker_line = None
        self._dragging = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Column selector
        self._col_selector = QComboBox()
        self._col_selector.currentTextChanged.connect(self._on_col_changed)
        layout.addWidget(self._col_selector)

        if _HAS_MPL:
            self._fig = Figure(figsize=(3.5, 2.2), dpi=80, facecolor="#f5f5f5")
            self._canvas = FigureCanvasQTAgg(self._fig)
            self._ax = self._fig.add_subplot(111)
            self._fig.tight_layout(pad=0.8)
            layout.addWidget(self._canvas)

            self._canvas.mpl_connect("button_press_event", self._mpl_press)
            self._canvas.mpl_connect("motion_notify_event", self._mpl_move)
            self._canvas.mpl_connect("button_release_event", self._mpl_release)
        else:
            lbl = QLabel(
                "matplotlib not found.\npip install matplotlib\nto enable the chart."
            )
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            layout.addWidget(lbl)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_global_data(self, data: list[dict], keep_column: str | None = None):
        """
        Replace the series. *keep_column* survives a simulation switch, if the
        new data still has it - otherwise the selection silently snaps back to
        mean_density_vpk on every switch.
        """
        self._global_data = data
        if not data:
            return
        cols = [k for k in data[0] if k != "datetime"]
        self._col_selector.blockSignals(True)
        self._col_selector.clear()
        self._col_selector.addItems(cols)
        preferred = keep_column if keep_column in cols else "mean_density_vpk"
        self._current_col = (
            preferred if preferred in cols else (cols[0] if cols else "")
        )
        if self._current_col:
            self._col_selector.setCurrentText(self._current_col)
        self._col_selector.blockSignals(False)
        self._redraw()

    def set_current_index(self, idx: int):
        self._current_idx = idx
        self._update_marker()

    def current_column(self) -> str:
        """The metric currently plotted, so a caller can restore it later."""
        return self._current_col

    def current_series(self) -> tuple[str, list[float]] | None:
        """
        (column, one value per timestep) for whatever the chart is showing.

        Used to redraw the same curve into recorded video. The video panel is
        re-rendered from the data rather than grabbed from this widget because
        _update_marker ends in draw_idle(), which defers to the next event-loop
        turn - a grab taken inside a frame capture would carry a stale marker.
        """
        if not self._global_data or not self._current_col:
            return None
        col = self._current_col
        return col, [float(d.get(col, 0) or 0.0) for d in self._global_data]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_col_changed(self, text: str):
        self._current_col = text
        self._redraw()

    def _redraw(self):
        if not _HAS_MPL or not self._global_data or not self._current_col:
            return
        ax = self._ax
        ax.clear()
        self._marker_line = None

        ys = [d.get(self._current_col, 0) for d in self._global_data]
        xs = list(range(len(ys)))

        ax.plot(xs, ys, color="#3a7ebf", linewidth=1.0)
        ax.set_xlabel("Time step", fontsize=6)
        ax.set_ylabel(self._current_col, fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlim(0, max(1, len(xs) - 1))

        if 0 <= self._current_idx < len(xs):
            self._marker_line = ax.axvline(
                xs[self._current_idx], color="#e63946", linewidth=1.2
            )

        self._fig.tight_layout(pad=0.6)
        self._canvas.draw()

    def _update_marker(self):
        if not _HAS_MPL or not self._global_data:
            return
        if self._marker_line is not None:
            try:
                self._marker_line.remove()
            except Exception:  # noqa: BLE001, S110 - best-effort cleanup of a
                # stale artist; any failure here just means it is already gone
                pass
            self._marker_line = None

        xs = list(range(len(self._global_data)))
        if 0 <= self._current_idx < len(xs):
            self._marker_line = self._ax.axvline(
                xs[self._current_idx], color="#e63946", linewidth=1.2
            )
        self._canvas.draw_idle()

    def _chart_x_to_index(self, event) -> int | None:
        if event.inaxes != self._ax or event.xdata is None:
            return None
        n = len(self._global_data)
        return max(0, min(n - 1, round(event.xdata)))

    def _mpl_press(self, event):
        idx = self._chart_x_to_index(event)
        if idx is not None:
            self._dragging = True
            self.time_index_changed.emit(idx)

    def _mpl_move(self, event):
        if self._dragging:
            idx = self._chart_x_to_index(event)
            if idx is not None:
                self.time_index_changed.emit(idx)

    def _mpl_release(self, _event):
        self._dragging = False


# ── Simulation selector dialog ────────────────────────────────────────────────


class SimulationDialog(QDialog):
    def __init__(self, simulations: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Simulation")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Choose which simulation to visualize:"))

        self._combo = QComboBox()
        for sim in simulations:
            self._combo.addItem(f"{sim['name']}  (ID: {sim['id']})", userData=sim["id"])
        layout.addWidget(self._combo)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_id(self) -> int:
        return self._combo.currentData()


# ── Main window ───────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Road Network Visualizer")
        self.resize(1280, 800)

        # ── App state ──────────────────────────────────────────────────────
        self._edges: list[dict] = []
        self._datetimes: list = []  # one datetime per timestep
        self._values: dict[str, np.ndarray] = {}  # key -> (T, n_edges) float32 array
        self._obs_domains: dict[str, tuple] = {}
        self._jam: np.ndarray | None = None  # (n_edges,) jam density in vpk
        self._vehicle_length: float = float("nan")  # recovered from the data
        self._vehicle_length_source: str = ""
        self._global_data: list[dict] = []
        self._current_idx: int = 0
        self._selected_obs: str = DEFAULT_OBSERVABLE
        self._is_playing: bool = False
        self._highlighted_edge: dict | None = None

        # ── Loaded file / simulation ───────────────────────────────────────
        self._db_path: Path | None = None
        self._simulations: list[dict] = []
        self._sim_id: int | None = None
        # The vehicle length is measured over every simulation in a file, so it
        # is cached here and the second scan skipped on a switch.
        self._file_vehicle_length: float | None = None
        self._sim_loader: SimulationLoader | None = None
        self._sim_thread: QThread | None = None
        self._sim_load_progress: QProgressDialog | None = None
        self._sim_reuse_geometry: bool = False

        # ── Recording state ────────────────────────────────────────────────
        # Chosen window; None means "whichever end of the timeline".
        self._in_idx: int | None = None
        self._out_idx: int | None = None
        # The recorder is held here for its whole run: dropping the last
        # reference would let it be collected and its timer would stop.
        self._recorder: VideoRecorder | None = None
        self._record_progress: QProgressDialog | None = None
        self._record_saved_state: tuple[bool, int] | None = None
        self._overlays: OverlayCompositor | None = None

        # Playback timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._playback_tick)

        # ── Build UI ───────────────────────────────────────────────────────
        self._build_toolbar()
        self._build_central()
        self._build_bottom_bar()
        self._set_data_loaded(False)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setStyleSheet("QToolBar { spacing: 6px; padding: 4px; }")
        self.addToolBar(tb)

        # Load DB
        act_load = QAction("📂  Load Database", self)
        act_load.setToolTip("Open a simulation SQLite database")
        act_load.triggered.connect(self._load_db)
        tb.addAction(act_load)

        # Simulation selector — switches run in place, keeping the map view
        tb.addWidget(QLabel(" Simulation: "))
        self._sim_combo = QComboBox()
        self._sim_combo.setMinimumWidth(190)
        self._sim_combo.setToolTip(
            "Switch simulation without moving the map or losing the time position"
        )
        self._sim_combo.currentIndexChanged.connect(self._on_sim_combo_changed)
        tb.addWidget(self._sim_combo)
        tb.addSeparator()

        # Tile style
        tb.addWidget(QLabel(" Tiles: "))
        self._tile_combo = QComboBox()
        self._tile_combo.addItems(["Inverted (dark)", "Grayscale", "Normal"])
        self._tile_combo.currentIndexChanged.connect(self._on_tile_filter_changed)
        tb.addWidget(self._tile_combo)
        tb.addSeparator()

        # Observable selector
        tb.addWidget(QLabel(" Color by: "))
        self._obs_combo = QComboBox()
        for key, cfg in EDGE_OBSERVABLE_CONFIG.items():
            self._obs_combo.addItem(cfg["label"], userData=key)
        default_idx = list(EDGE_OBSERVABLE_CONFIG).index(self._selected_obs)
        self._obs_combo.setCurrentIndex(default_idx)
        self._obs_combo.currentIndexChanged.connect(self._on_obs_changed)
        tb.addWidget(self._obs_combo)

        # Upper end of the colour scale for the normalised density, as a
        # fraction of each edge's own capacity: the occupancy that reads red.
        # Only meaningful for "Density (normalized)".
        self._norm_max_label = tb.addWidget(QLabel(" Scale max: "))
        self._norm_max_spin = QDoubleSpinBox()
        self._norm_max_spin.setDecimals(2)
        self._norm_max_spin.setRange(0.01, 1.0)
        self._norm_max_spin.setSingleStep(0.05)
        self._norm_max_spin.setValue(DENSITY_NORM_SATURATION)
        self._norm_max_spin.setToolTip(
            "Occupancy that maps to red; yellow falls at half of it "
            "(1.00 = bumper to bumper)"
        )
        self._norm_max_spin.valueChanged.connect(self._on_norm_max_changed)
        self._norm_max_action = tb.addWidget(self._norm_max_spin)
        self._update_norm_max_visibility()
        tb.addSeparator()

        # Screenshot
        act_shot = QAction("📷  Screenshot", self)
        act_shot.setToolTip("Save a PNG screenshot of the map")
        act_shot.triggered.connect(self._take_screenshot)
        tb.addAction(act_shot)

    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)

        # ── Left: map ──────────────────────────────────────────────────────
        self._map = MapWidget()
        self._map.edge_clicked.connect(self._on_edge_clicked)
        splitter.addWidget(self._map)

        # ── Right: side panel ──────────────────────────────────────────────
        side = QWidget()
        side.setFixedWidth(280)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 4, 4, 4)
        side_layout.setSpacing(6)

        # Search
        search_box = QGroupBox("Search")
        sl = QVBoxLayout(search_box)
        sl.setSpacing(4)

        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Edge ID:"))
        self._edge_search = QLineEdit()
        self._edge_search.setPlaceholderText("e.g. 42")
        self._edge_search.returnPressed.connect(self._search_edge)
        edge_row.addWidget(self._edge_search)
        self._edge_search_btn = QPushButton("Go")
        self._edge_search_btn.setFixedWidth(32)
        self._edge_search_btn.clicked.connect(self._search_edge)
        edge_row.addWidget(self._edge_search_btn)
        sl.addLayout(edge_row)

        node_row = QHBoxLayout()
        node_row.addWidget(QLabel("Node ID:"))
        self._node_search = QLineEdit()
        self._node_search.setPlaceholderText("e.g. 123")
        self._node_search.returnPressed.connect(self._search_node)
        node_row.addWidget(self._node_search)
        self._node_search_btn = QPushButton("Go")
        self._node_search_btn.setFixedWidth(32)
        self._node_search_btn.clicked.connect(self._search_node)
        node_row.addWidget(self._node_search_btn)
        sl.addLayout(node_row)

        btn_row = QHBoxLayout()
        self._inverse_btn = QPushButton("Inverse Edge")
        self._inverse_btn.setToolTip("Select the reverse direction of the current edge")
        self._inverse_btn.clicked.connect(self._inverse_edge)
        btn_row.addWidget(self._inverse_btn)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_selection)
        btn_row.addWidget(self._clear_btn)
        sl.addLayout(btn_row)

        side_layout.addWidget(search_box)

        # Edge info
        info_box = QGroupBox("Edge Info")
        il = QVBoxLayout(info_box)
        self._info_label = QLabel("No edge selected.")
        self._info_label.setWordWrap(True)
        self._info_label.setTextFormat(Qt.RichText)
        self._info_label.setAlignment(Qt.AlignTop)
        self._info_label.setStyleSheet("font-size: 11px;")
        il.addWidget(self._info_label)
        side_layout.addWidget(info_box)

        # Legend
        legend_box = QGroupBox("Legend")
        ll = QVBoxLayout(legend_box)
        self._legend = LegendWidget()
        ll.addWidget(self._legend)
        side_layout.addWidget(legend_box)

        # Chart
        chart_box = QGroupBox("Statistics Chart")
        cl = QVBoxLayout(chart_box)
        cl.setContentsMargins(2, 2, 2, 2)
        self._chart = ChartWidget()
        self._chart.time_index_changed.connect(self._jump_to_index)
        cl.addWidget(self._chart)
        side_layout.addWidget(chart_box)

        side_layout.addStretch()
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # Wrap splitter in a container so we can stack the bottom bar below it
        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)
        vl.addWidget(splitter, stretch=1)

        self._bottom_bar = QWidget()
        self._bottom_bar.setFixedHeight(62)
        # self._bottom_bar.setStyleSheet(
        #     "background: #f0f0f0; border-top: 1px solid #ccc;")
        vl.addWidget(self._bottom_bar)

        self.setCentralWidget(container)

    def _build_bottom_bar(self):
        layout = QHBoxLayout(self._bottom_bar)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)

        # Play / pause
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(36, 36)
        self._play_btn.setToolTip("Play / pause animation")
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)

        # FPS
        layout.addWidget(QLabel("FPS:"))
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 60.0)
        self._fps_spin.setValue(2.0)
        self._fps_spin.setDecimals(1)
        self._fps_spin.setFixedWidth(58)
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        layout.addWidget(self._fps_spin)

        layout.addSpacing(6)

        # Recording window: mark in / out around the current frame
        self._in_btn = QPushButton("⟦")
        self._in_btn.setFixedSize(26, 26)
        self._in_btn.setToolTip("Start the recording at the current frame")
        self._in_btn.clicked.connect(self._set_range_in)
        layout.addWidget(self._in_btn)

        # Slider
        self._slider = RangeSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setSingleStep(1)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, stretch=1)

        self._out_btn = QPushButton("⟧")
        self._out_btn.setFixedSize(26, 26)
        self._out_btn.setToolTip("End the recording at the current frame")
        self._out_btn.clicked.connect(self._set_range_out)
        layout.addWidget(self._out_btn)

        self._clear_range_btn = QPushButton("✕")
        self._clear_range_btn.setFixedSize(26, 26)
        self._clear_range_btn.setToolTip(
            "Clear the recording window (record everything)"
        )
        self._clear_range_btn.clicked.connect(self._clear_range)
        layout.addWidget(self._clear_range_btn)

        layout.addSpacing(6)

        # Time label
        self._time_label = QLabel("—")
        self._time_label.setFixedWidth(130)
        self._time_label.setAlignment(Qt.AlignCenter)
        self._time_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        layout.addWidget(self._time_label)

        # Chosen window, then the button that records it
        self._range_label = QLabel("—")
        self._range_label.setMinimumWidth(190)
        self._range_label.setAlignment(Qt.AlignCenter)
        self._range_label.setStyleSheet("font-size: 10px; color: #555;")
        layout.addWidget(self._range_label)

        # Size of the chart panel burned into the video ("No chart" turns it off)
        self._chart_size_combo = QComboBox()
        self._chart_size_combo.addItems(list(CHART_SIZES))
        self._chart_size_combo.setCurrentText(DEFAULT_CHART_SIZE)
        self._chart_size_combo.setFixedWidth(92)
        self._chart_size_combo.setToolTip(
            "Size of the load chart drawn into the recorded video"
        )
        layout.addWidget(self._chart_size_combo)

        self._record_btn = QPushButton("⏺  Record")
        self._record_btn.setFixedHeight(28)
        self._record_btn.setToolTip("Record the selected time window to an MP4 video")
        self._record_btn.clicked.connect(self._record_video)
        layout.addWidget(self._record_btn)

    # ── Database loading ──────────────────────────────────────────────────────

    def _load_db(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Simulation Database",
            "",
            "SQLite databases (*.db *.sqlite *.sqlite3);;All files (*)",
        )
        if not path:
            return
        self._db_path = Path(path)

        # Open and validate
        try:
            conn = sqlite3.connect(path)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        except sqlite3.Error as exc:
            QMessageBox.critical(self, "Error", f"Cannot open database:\n{exc}")
            return

        required = {"edges", "road_data", "simulation_info"}
        missing = required - tables
        if missing:
            QMessageBox.critical(
                self,
                "Error",
                f"Database is missing required tables: {', '.join(sorted(missing))}",
            )
            return

        sims = get_simulations(conn)
        if not sims:
            QMessageBox.critical(self, "Error", "No simulations found in database.")
            return

        # Simulation selector
        if len(sims) == 1:
            sim_id = sims[0]["id"]
        else:
            dlg = SimulationDialog(sims, self)
            accepted = dlg.exec() == QDialog.Accepted
            sim_id = dlg.selected_id()
            # The dialog is parented to the window, so without this it lingers
            # as a dead child for the life of the app - one per Load Database.
            dlg.deleteLater()
            if not accepted:
                return

        # Geometry is shared by every simulation in the file (edges has no
        # simulation_id), so it is read once here and reused across switches.
        try:
            edges = load_edges(conn)
        except Exception as exc:  # noqa: BLE001 - malformed geometry/SQL, show it
            QMessageBox.critical(self, "Error", f"Failed to load edges:\n{exc}")
            return
        finally:
            conn.close()

        if not edges:
            QMessageBox.critical(self, "Error", "No edges found in database.")
            return

        self._edges = edges
        self._simulations = sims
        self._file_vehicle_length = None  # re-measured for a newly opened file
        self._populate_sim_combo(sim_id)
        self._start_simulation_load(sim_id, reuse_geometry=False)

    # ── Simulation switching ──────────────────────────────────────────────────

    def _populate_sim_combo(self, current_id: int | None):
        """Fill the toolbar selector without its own signal firing a switch."""
        self._sim_combo.blockSignals(True)
        self._sim_combo.clear()
        for sim in self._simulations:
            self._sim_combo.addItem(sim["name"], userData=sim["id"])
        if current_id is not None:
            idx = self._sim_combo.findData(current_id)
            if idx >= 0:
                self._sim_combo.setCurrentIndex(idx)
        self._sim_combo.blockSignals(False)

    @Slot(int)
    def _on_sim_combo_changed(self, _index: int):
        sim_id = self._sim_combo.currentData()
        if sim_id is None or sim_id == self._sim_id or self._sim_loader is not None:
            return
        self._start_simulation_load(sim_id, reuse_geometry=True)

    def _start_simulation_load(self, sim_id: int, *, reuse_geometry: bool):
        """
        Read one simulation on a worker thread, behind a cancellable dialog.

        The read is seconds long. On the GUI thread it would freeze the window -
        and, because Qt would never get an event-loop turn, whatever widget
        triggered the load would stay painted on screen until it finished.
        """
        if self._db_path is None or self._sim_loader is not None:
            return

        self._sim_load_progress = QProgressDialog(
            "Reading simulation data…", "Cancel", 0, 0, self
        )
        self._sim_load_progress.setWindowTitle("Loading")
        self._sim_load_progress.setWindowModality(Qt.WindowModal)
        self._sim_load_progress.setMinimumDuration(0)
        self._sim_load_progress.setAutoClose(False)
        self._sim_load_progress.setAutoReset(False)
        self._sim_load_progress.setValue(0)

        loader = SimulationLoader(
            self._db_path, sim_id, self._edges, self._file_vehicle_length
        )
        thread = QThread(self)
        loader.moveToThread(thread)
        thread.started.connect(loader.run)
        # Both slots must be bound methods of *this* object, not lambdas: a
        # lambda has no thread affinity, so Qt would run it on the worker
        # thread, where quitting and waiting for that same thread deadlocks.
        loader.progress.connect(self._on_sim_load_progress)
        loader.done.connect(self._on_sim_loaded)
        # Cancel is called directly rather than connected to loader.cancel: the
        # loader lives on the worker thread, which is deep in the read loop and
        # would not reach its event loop to receive a queued call. Setting the
        # loader's threading.Event from here is the thread-safe way in.
        self._sim_load_progress.canceled.connect(self._cancel_sim_load)

        self._sim_reuse_geometry = reuse_geometry
        self._sim_loader = loader
        self._sim_thread = thread
        thread.start()

    @Slot()
    def _cancel_sim_load(self):
        if self._sim_loader is not None:
            self._sim_loader.cancel()

    @Slot(int)
    def _on_sim_load_progress(self, rows: int):
        if self._sim_load_progress is not None:
            self._sim_load_progress.setLabelText(f"Read {rows / 1e6:.1f}M rows…")

    @Slot(object, str)
    def _on_sim_loaded(self, result, error: str):
        reuse_geometry = self._sim_reuse_geometry
        if self._sim_load_progress is not None:
            self._sim_load_progress.reset()
            self._sim_load_progress.deleteLater()
            self._sim_load_progress = None

        thread, self._sim_thread = self._sim_thread, None
        loader, self._sim_loader = self._sim_loader, None
        if thread is not None:
            thread.quit()
            thread.wait(5000)
            thread.deleteLater()
        if loader is not None:
            loader.deleteLater()

        if result is None:
            # Cancelled (error "") or failed: the current simulation is still
            # loaded and untouched, so only the selector has to be put back.
            self._populate_sim_combo(self._sim_id)
            if error:
                QMessageBox.critical(self, "Error", f"Failed to load data:\n{error}")
            else:
                self.statusBar().showMessage("Loading cancelled", 4000)
            return

        bundle = result["bundle"]
        if not bundle["datetimes"]:
            self._populate_sim_combo(self._sim_id)
            QMessageBox.critical(
                self,
                "Error",
                f"No road_data found for simulation ID {result['sim_id']}.",
            )
            return

        self._apply_simulation(
            result["sim_id"],
            bundle,
            result["global_data"],
            reuse_geometry=reuse_geometry,
        )

    def _apply_simulation(
        self,
        sim_id: int,
        bundle: dict,
        global_data: list[dict],
        *,
        reuse_geometry: bool,
    ):
        """
        Install a freshly loaded simulation.

        With reuse_geometry the map is not touched at all - no set_edges, no
        fit_bounds - so zoom and centre stay exactly where the user left them,
        and the time position, recording marks, selected edge and colour
        settings are carried across too.
        """
        # What to carry over, captured before anything is replaced
        old_datetimes = self._datetimes
        old_idx = self._current_idx
        old_in, old_out = self._in_idx, self._out_idx
        old_edge = self._highlighted_edge
        old_col = self._chart.current_column()

        self._sim_id = sim_id
        self._datetimes = bundle["datetimes"]
        self._values = bundle["values"]
        self._obs_domains = bundle["domains"]
        self._jam = bundle.get("jam_density")
        self._vehicle_length = bundle.get("vehicle_length_m", float("nan"))
        self._vehicle_length_source = bundle.get("vehicle_length_source", "")
        self._obs_domains["density_norm"] = (0.0, self._norm_max_spin.value())
        self._global_data = global_data
        # Measured over every simulation in the file, so it holds for the next
        # switch too and that second full scan can be skipped.
        if self._file_vehicle_length is None and math.isfinite(self._vehicle_length):
            self._file_vehicle_length = self._vehicle_length

        n = len(self._datetimes)

        if not reuse_geometry:
            self._current_idx = 0
            self._highlighted_edge = None
            self._map.set_edges(self._edges)
            self._map.highlighted_edge_id = None
            self._map.highlighted_node = None
            # Centre map on median geometry coordinate
            all_lats = [lat for e in self._edges for lon, lat in e["geometry"]]
            all_lons = [lon for e in self._edges for lon, lat in e["geometry"]]
            if all_lats:
                self._map.fit_bounds(
                    min(all_lats), min(all_lons), max(all_lats), max(all_lons)
                )
            self._in_idx = None
            self._out_idx = None
            new_idx = 0
        else:
            # Same moment of the day rather than the same row: the runs may be
            # different days, and may not even share a timestep grid.
            new_idx = _nearest_time_index(old_datetimes, old_idx, self._datetimes)
            self._in_idx = None if old_in is None else min(old_in, n - 1)
            self._out_idx = None if old_out is None else min(old_out, n - 1)
            self._highlighted_edge = old_edge

        self._slider.blockSignals(True)
        self._slider.setMaximum(n - 1)
        self._slider.setValue(new_idx)
        self._slider.blockSignals(False)
        self._update_range_ui()

        self._chart.set_global_data(global_data, keep_column=old_col)
        self._refresh_legend()

        self._apply_timestep(new_idx)
        self._set_data_loaded(True)
        name = next(
            (s["name"] for s in self._simulations if s["id"] == sim_id), str(sim_id)
        )
        self.statusBar().showMessage(
            f"{name} · {len(self._edges)} edges · {n} timesteps · vehicle length "
            f"{self._vehicle_length:.2f} m ({self._vehicle_length_source})",
            15000,
        )

    # ── Visualization update ──────────────────────────────────────────────────

    def _apply_timestep(self, idx: int):
        """Compute colours + densities for timestep *idx* on demand (O(n_edges))."""
        self._current_idx = idx

        key = self._selected_obs
        colors = (
            edge_colors_for_timestep(
                key, idx, self._values, self._obs_domains, jam=self._jam
            )
            if self._values
            else []
        )
        occupancy = (
            observable_row("density_norm", idx, self._values, self._jam).tolist()
            if self._values
            else []
        )

        self._map.set_edge_colors(colors)
        self._map.set_edge_occupancy(occupancy)

        # Update time label
        dt = self._datetimes[idx]
        self._time_label.setText(dt.strftime("%Y-%m-%d %H:%M"))

        # Update chart marker
        self._chart.set_current_index(idx)

        # Refresh edge info if one is selected
        if self._highlighted_edge:
            self._show_edge_info(self._highlighted_edge, idx)

    def _legend_spec(self) -> tuple[str, tuple[float, float], bool]:
        """
        What the legend currently says: (label, domain, reversed).

        One place decides this, so the side-panel legend and the one burned into
        a recording always agree - including a "Scale max" the user has changed,
        which lives in _obs_domains.
        """
        key = self._selected_obs
        cfg = EDGE_OBSERVABLE_CONFIG.get(key, {})
        return (
            cfg.get("label", key),
            self._obs_domains.get(key, (0.0, 1.0)),
            cfg.get("reverseColorScale", False),
        )

    def _refresh_legend(self):
        _label, domain, rev = self._legend_spec()
        self._legend.set_observable(self._selected_obs, domain, rev)

    # ── Slider / playback ─────────────────────────────────────────────────────

    @Slot(int)
    def _on_slider_changed(self, value: int):
        if self._datetimes:
            self._apply_timestep(value)

    @Slot(int)
    def _jump_to_index(self, idx: int):
        """Called when the user clicks/drags the chart."""
        self._slider.setValue(idx)

    def _toggle_play(self):
        self._is_playing = not self._is_playing
        self._play_btn.setText("⏸" if self._is_playing else "▶")
        if self._is_playing:
            fps = self._fps_spin.value()
            self._timer.start(max(1, int(1000 / fps)))
        else:
            self._timer.stop()

    @Slot()
    def _playback_tick(self):
        n = len(self._datetimes)
        if n == 0:
            return
        next_idx = (self._current_idx + 1) % n
        self._slider.setValue(next_idx)

    @Slot(float)
    def _on_fps_changed(self, fps: float):
        if self._is_playing:
            self._timer.setInterval(max(1, int(1000 / fps)))
        self._update_range_ui()  # the label quotes the resulting video duration

    # ── Recording window ──────────────────────────────────────────────────────

    def _record_range(self) -> tuple[int, int]:
        """The window to record, as inclusive timestep indices."""
        last = max(0, len(self._datetimes) - 1)
        lo = 0 if self._in_idx is None else min(self._in_idx, last)
        hi = last if self._out_idx is None else min(self._out_idx, last)
        return (lo, hi) if lo <= hi else (hi, lo)

    @Slot()
    def _set_range_in(self):
        if not self._datetimes:
            return
        self._in_idx = self._current_idx
        # Dragging the start past the end takes the end with it, rather than
        # refusing the click and leaving the user to guess why.
        if self._out_idx is not None and self._out_idx < self._in_idx:
            self._out_idx = self._in_idx
        self._update_range_ui()

    @Slot()
    def _set_range_out(self):
        if not self._datetimes:
            return
        self._out_idx = self._current_idx
        if self._in_idx is not None and self._in_idx > self._out_idx:
            self._in_idx = self._out_idx
        self._update_range_ui()

    @Slot()
    def _clear_range(self):
        self._in_idx = None
        self._out_idx = None
        self._update_range_ui()

    def _update_range_ui(self):
        if not self._datetimes:
            self._slider.set_marked_range(None, None)
            self._range_label.setText("—")
            return
        lo, hi = self._record_range()
        self._slider.set_marked_range(lo, hi)
        n = hi - lo + 1
        fps = self._fps_spin.value()
        span = f"{self._datetimes[lo]:%H:%M} → {self._datetimes[hi]:%H:%M}"
        whole = self._in_idx is None and self._out_idx is None
        prefix = "full range · " if whole else ""
        self._range_label.setText(f"{prefix}{span} · {n} frames · {n / fps:.1f} s")

    # ── Toolbar handlers ──────────────────────────────────────────────────────

    @Slot(int)
    def _on_tile_filter_changed(self, index: int):
        modes = ["invert", "gray", "normal"]
        self._map.set_tile_filter(modes[index])

    @Slot(int)
    def _on_obs_changed(self, _index: int):
        key = self._obs_combo.currentData()
        if key and key != self._selected_obs:
            self._selected_obs = key
            self._update_norm_max_visibility()
            self._refresh_legend()
            if self._datetimes:
                self._apply_timestep(self._current_idx)

    def _update_norm_max_visibility(self):
        show = self._selected_obs == "density_norm"
        self._norm_max_label.setVisible(show)
        self._norm_max_action.setVisible(show)

    @Slot(float)
    def _on_norm_max_changed(self, value: float):
        self._obs_domains["density_norm"] = (0.0, value)
        self._refresh_legend()
        if self._datetimes and self._selected_obs == "density_norm":
            self._apply_timestep(self._current_idx)

    def _take_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "screenshot.png",
            "PNG images (*.png);;All files (*)",
        )
        if not path:
            return
        pixmap = self._map.grab()
        if pixmap.save(path):
            self.statusBar().showMessage(f"Screenshot saved to {path}", 4000)
        else:
            QMessageBox.warning(self, "Error", f"Could not save screenshot to:\n{path}")

    # ── Video recording ───────────────────────────────────────────────────────

    @Slot()
    def _record_video(self):
        if not self._datetimes or self._recorder is not None:
            return
        if ffmpeg_path() is None:
            QMessageBox.critical(
                self,
                "ffmpeg not found",
                "Recording needs the ffmpeg command-line tool, which is not on "
                "your PATH.\n\nInstall it (e.g. 'sudo apt install ffmpeg') and "
                "try again.",
            )
            return

        lo, hi = self._record_range()
        indices = list(range(lo, hi + 1))
        stem = (
            self._db_path.with_suffix("") if self._db_path else Path.home() / "density"
        )
        suggested = f"{stem}_{self._selected_obs}.mp4"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Video", suggested, "MP4 video (*.mp4);;All files (*)"
        )
        if not path:
            return

        self._begin_record_state()

        # Panels for the burned-in overlay, built from what the UI is showing
        # right now. Rendering happens on the first frame, when the real pixel
        # size is known.
        label, domain, rev = self._legend_spec()
        series = self._chart.current_series()
        self._overlays = OverlayCompositor(
            legend_label=label,
            legend_domain=domain,
            legend_reversed=rev,
            chart_title=pretty_metric(series[0]) if series else "",
            chart_values=series[1] if series else (),
            chart_fraction=CHART_SIZES.get(self._chart_size_combo.currentText(), 0.0),
            highlight=(lo, hi),
        )

        self._record_progress = QProgressDialog(
            "Loading map tiles…", "Cancel", 0, len(indices), self
        )
        self._record_progress.setWindowTitle("Recording")
        self._record_progress.setWindowModality(Qt.WindowModal)
        self._record_progress.setMinimumDuration(0)
        self._record_progress.setAutoClose(False)
        self._record_progress.setAutoReset(False)
        self._record_progress.setValue(0)

        self._recorder = VideoRecorder(
            indices=indices,
            sink=FFmpegSink(Path(path)),
            fps=self._fps_spin.value(),
            apply_frame=self._seek_for_record,
            capture=self._capture_record_frame,
            tiles_pending=self._map.pending_tile_count,
            request_tiles=self._map.request_visible_tiles,
            parent=self,
        )
        self._record_progress.canceled.connect(self._recorder.cancel)
        self._recorder.progress.connect(self._on_record_progress)
        self._recorder.finished.connect(self._on_record_finished)
        self._recorder.start()

    def _capture_record_frame(self) -> QImage:
        """One video frame: the map grab with the overlay panels burned in."""
        img = self._map.grab().toImage()
        # A HiDPI grab carries devicePixelRatio 2, which would make QPainter
        # read our device-pixel coordinates as logical ones and draw every
        # panel at half scale.
        img.setDevicePixelRatio(1.0)
        if img.format() != QImage.Format_RGB32:
            img = img.convertToFormat(QImage.Format_RGB32)

        overlays = self._overlays
        if overlays is None:
            return img
        if overlays.size is None:
            overlays.prepare(even_size(img.size()))
        # Normalise to the pinned size *before* compositing, so that resizing
        # the window mid-recording cannot push a panel outside the frame and
        # have video_export's own fit() crop it.
        img = fit(img, overlays.size)
        overlays.composite(img, self._current_idx)
        return img

    def _seek_for_record(self, idx: int):
        """Put slider, chart marker and map on timestep *idx*."""
        if self._slider.value() == idx:
            # valueChanged does not fire for an unchanged value, so the first
            # frame (and a one-frame window) would never be drawn.
            self._apply_timestep(idx)
        else:
            self._slider.setValue(idx)  # → _on_slider_changed → _apply_timestep

    @Slot(int, int)
    def _on_record_progress(self, done: int, total: int):
        if self._record_progress is not None:
            self._record_progress.setLabelText(f"Encoding frame {done} of {total}…")
            self._record_progress.setValue(done)

    @Slot(bool, str)
    def _on_record_finished(self, ok: bool, message: str):
        if self._record_progress is not None:
            self._record_progress.reset()
            self._record_progress.deleteLater()
            self._record_progress = None

        recorder, self._recorder = self._recorder, None
        if recorder is not None:
            recorder.deleteLater()
        self._overlays = None  # drop the rendered panels
        self._end_record_state()

        if ok:
            self.statusBar().showMessage(f"Video saved to {message}", 8000)
        elif message:
            QMessageBox.warning(self, "Recording failed", message)
        else:
            self.statusBar().showMessage("Recording cancelled", 4000)

    def _begin_record_state(self):
        self._record_saved_state = (self._is_playing, self._current_idx)
        if self._is_playing:
            self._toggle_play()
        self._record_btn.setEnabled(False)

    def _end_record_state(self):
        self._record_btn.setEnabled(bool(self._datetimes))
        saved, self._record_saved_state = self._record_saved_state, None
        if saved is None or not self._datetimes:
            return
        was_playing, idx = saved
        self._seek_for_record(min(idx, len(self._datetimes) - 1))
        if was_playing and not self._is_playing:
            self._toggle_play()

    # ── Search ────────────────────────────────────────────────────────────────

    def _search_edge(self):
        raw = self._edge_search.text().strip()
        if not raw:
            return
        # Try numeric ID first, then string match
        edge = next(
            (e for e in self._edges if str(e["id"]) == raw or e["id"] == raw), None
        )
        if edge is None:
            self._info_label.setText(
                f"<span style='color:red'>Edge '{raw}' not found.</span>"
            )
            return
        idx = self._edges.index(edge)
        self._select_edge(edge, idx)
        self._zoom_to_edge(edge)

    def _search_node(self):
        raw = self._node_search.text().strip()
        if not raw:
            return

        # Find first edge where source or target matches
        edge = next(
            (
                e
                for e in self._edges
                if str(e.get("source", "")) == raw or str(e.get("target", "")) == raw
            ),
            None,
        )
        if edge is None:
            self._info_label.setText(
                f"<span style='color:red'>Node '{raw}' not found.</span>"
            )
            return

        is_source = str(edge.get("source", "")) == raw
        geom = edge["geometry"]
        if geom:
            lon, lat = geom[0] if is_source else geom[-1]
            self._map.highlighted_node = (lon, lat)
            self._map.set_center(lat, lon, zoom=min(18, self._map.zoom + 2))
            self._info_label.setText(
                f"<b>Node ID:</b> {raw}<br><b>Position:</b> ({lon:.6f}, {lat:.6f})"
            )
        self._map.highlighted_edge_id = None
        self._highlighted_edge = None
        self._inverse_btn.setEnabled(False)

    def _inverse_edge(self):
        """Select the opposing direction of the currently highlighted edge."""
        if not self._highlighted_edge:
            return
        src = self._highlighted_edge.get("source")
        tgt = self._highlighted_edge.get("target")
        inv = next(
            (
                e
                for e in self._edges
                if e.get("source") == tgt and e.get("target") == src
            ),
            None,
        )
        if inv is None:
            QMessageBox.information(
                self, "Not found", f"No inverse edge from '{tgt}' to '{src}' found."
            )
            return
        self._select_edge(inv, self._edges.index(inv))
        self._zoom_to_edge(inv)

    def _clear_selection(self):
        self._highlighted_edge = None
        self._map.highlighted_edge_id = None
        self._map.highlighted_node = None
        self._map.update()
        self._info_label.setText("No edge selected.")
        self._inverse_btn.setEnabled(False)
        self._edge_search.clear()
        self._node_search.clear()

    # ── Edge click (from MapWidget signal) ────────────────────────────────────

    @Slot(dict, int)
    def _on_edge_clicked(self, edge: dict, idx: int):
        self._select_edge(edge, idx)
        self._zoom_to_edge(edge)

    def _select_edge(self, edge: dict, edge_idx: int):
        self._highlighted_edge = edge
        self._map.highlighted_edge_id = edge["id"]
        self._map.highlighted_node = None
        self._map.update()
        self._show_edge_info(edge, self._current_idx)
        self._inverse_btn.setEnabled(True)

    def _show_edge_info(self, edge: dict, ts_idx: int):
        density = "N/A"
        norm_density = "N/A"
        capacity = "N/A"
        if math.isfinite(self._vehicle_length) and edge.get("length"):
            lane_m = float(edge["length"]) * max(1, int(edge.get("nlanes") or 1))
            capacity = f"{max(1, math.ceil(lane_m / self._vehicle_length))} vehicles"
        dens_arr = self._values.get("density") if self._values else None
        if dens_arr is not None and ts_idx < len(dens_arr):
            try:
                edge_pos = self._edges.index(edge)
                d = dens_arr[ts_idx][edge_pos]
                density = f"{d:.2f} veh/km"
                if self._jam is not None and edge_pos < len(self._jam):
                    jam = float(self._jam[edge_pos])
                    if jam > 0:
                        norm_density = f"{d / jam:.3f}  ({d / jam * 100:.1f} %)"
            except ValueError, IndexError:
                pass

        self._info_label.setText(
            f"<b>Edge ID:</b> {edge.get('id', 'N/A')}<br>"
            f"<b>Source:</b> {edge.get('source', 'N/A')}<br>"
            f"<b>Target:</b> {edge.get('target', 'N/A')}<br>"
            f"<b>Name:</b> {edge.get('name', 'N/A')}<br>"
            f"<b>Max Speed:</b> {edge.get('maxspeed', 'N/A')}<br>"
            f"<b>Lanes:</b> {edge.get('nlanes', 'N/A')}<br>"
            f"<b>Density:</b> {density}<br>"
            f"<b>Norm. Density:</b> {norm_density}<br>"
            f"<b>Capacity:</b> {capacity}<br>"
            f"<b>Coil Code:</b> {edge.get('coilcode', 'N/A')}"
        )

    def _zoom_to_edge(self, edge: dict):
        geom = edge.get("geometry", [])
        if not geom:
            return
        lats = [lat for lon, lat in geom]
        lons = [lon for lon, lat in geom]
        self._map.fit_bounds(min(lats), min(lons), max(lats), max(lons))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_data_loaded(self, loaded: bool):
        """Enable/disable controls that require data to be present."""
        # Only switchable with a file open, more than one run in it, and no
        # load already running.
        self._sim_combo.setEnabled(loaded and len(self._simulations) > 1)
        self._play_btn.setEnabled(loaded)
        self._slider.setEnabled(loaded)
        self._obs_combo.setEnabled(loaded)
        self._inverse_btn.setEnabled(False)  # enabled per-selection
        self._clear_btn.setEnabled(loaded)
        self._edge_search_btn.setEnabled(loaded)
        self._node_search_btn.setEnabled(loaded)
        self._in_btn.setEnabled(loaded)
        self._out_btn.setEnabled(loaded)
        self._clear_range_btn.setEnabled(loaded)
        self._chart_size_combo.setEnabled(loaded)
        self._record_btn.setEnabled(loaded)
        if not loaded:
            self._time_label.setText("—")
            self._range_label.setText("—")

    def closeEvent(self, event):
        # A recording in flight has to be torn down here: once the event loop
        # stops there is no next tick to notice the cancel, and ffmpeg would
        # outlive the app holding a half-written file.
        if self._recorder is not None and self._recorder.is_active():
            self._recorder.cancel()
            self._recorder.finish_now()
        # A simulation read in flight has to be stopped too, or the worker
        # thread outlives the window it would emit into.
        if self._sim_loader is not None:
            self._sim_loader.cancel()
        if self._sim_thread is not None:
            self._sim_thread.quit()
            self._sim_thread.wait(5000)
        # Stop playback and tile loader threads cleanly
        self._timer.stop()
        self._map._pool.waitForDone(2000)
        super().closeEvent(event)
