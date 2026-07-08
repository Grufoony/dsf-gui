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

import sqlite3
from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QPainter
from PySide6.QtWidgets import (
    QApplication,
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
    QPushButton,
    QSlider,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .map_widget import MapWidget
from .database import (
    EDGE_OBSERVABLE_CONFIG,
    MAX_DENSITY,
    get_simulations,
    load_edges,
    load_road_data,
    load_global_data,
    precompute_all_colors,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# ── Legend widget ─────────────────────────────────────────────────────────────


class LegendWidget(QWidget):
    """Draws a green→yellow→red gradient bar with domain labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._label = "Density"
        self._domain = (0.0, MAX_DENSITY)
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

        # Gradient bar  (y = 18..35)
        for i in range(w):
            t = i / max(1, w - 1)
            if self._reversed:
                t = 1.0 - t
            if t <= 0.5:
                s = t * 2
                r, g = int(s * 255), int(128 + s * 127)
            else:
                s = (t - 0.5) * 2
                r, g = 255, int((1.0 - s) * 255)
            p.setPen(QColor(r, g, 0))
            p.drawLine(i, 18, i, 35)

        # Border
        p.setPen(QColor(120, 120, 120))
        p.setBrush(Qt.NoBrush)
        p.drawRect(0, 18, w - 1, 17)

        # Labels
        p.setFont(QFont("Arial", 7))
        p.setPen(QColor(0, 0, 0))
        dmin, dmax = self._domain
        dmid = (dmin + dmax) / 2

        def fmt(v: float) -> str:
            if abs(v) >= 1000:
                return f"{v:.0f}"
            if abs(v) >= 100:
                return f"{v:.0f}"
            if abs(v) >= 10:
                return f"{v:.1f}"
            return f"{v:.2f}"

        fm = p.fontMetrics()
        min_s = fmt(dmin)
        mid_s = fmt(dmid)
        max_s = fmt(dmax)
        mid_x = (w - fm.horizontalAdvance(mid_s)) // 2
        max_x = w - fm.horizontalAdvance(max_s)
        p.drawText(2, 54, min_s)
        p.drawText(mid_x, 54, mid_s)
        p.drawText(max_x, 54, max_s)
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

    def set_global_data(self, data: list[dict]):
        self._global_data = data
        if not data:
            return
        cols = [k for k in data[0] if k != "datetime"]
        self._col_selector.blockSignals(True)
        self._col_selector.clear()
        self._col_selector.addItems(cols)
        preferred = "mean_density_vpk"
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
            except Exception:
                pass
            self._marker_line = None

        xs = list(range(len(self._global_data)))
        if 0 <= self._current_idx < len(xs):
            self._marker_line = self._ax.axvline(
                xs[self._current_idx], color="#e63946", linewidth=1.2
            )
        self._canvas.draw_idle()

    def _chart_x_to_index(self, event) -> Optional[int]:
        if event.inaxes != self._ax or event.xdata is None:
            return None
        n = len(self._global_data)
        return max(0, min(n - 1, int(round(event.xdata))))

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
        self._densities: list[dict] = []  # [{datetime, densities:[]}]
        self._obs_data: dict[str, list[dict]] = {}
        self._obs_domains: dict[str, tuple] = {}
        self._global_data: list[dict] = []
        self._precomputed: dict[str, list[list[QColor]]] = {}
        self._current_idx: int = 0
        self._selected_obs: str = "density"
        self._is_playing: bool = False
        self._highlighted_edge: Optional[dict] = None

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
        self._obs_combo.currentIndexChanged.connect(self._on_obs_changed)
        tb.addWidget(self._obs_combo)
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
        self._fps_spin.setValue(10.0)
        self._fps_spin.setDecimals(1)
        self._fps_spin.setFixedWidth(58)
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        layout.addWidget(self._fps_spin)

        layout.addSpacing(6)

        # Slider
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setSingleStep(1)
        self._slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._slider, stretch=1)

        layout.addSpacing(6)

        # Time label
        self._time_label = QLabel("—")
        self._time_label.setFixedWidth(130)
        self._time_label.setAlignment(Qt.AlignCenter)
        self._time_label.setStyleSheet("font-size: 11px; font-weight: bold;")
        layout.addWidget(self._time_label)

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

        # Open and validate
        try:
            conn = sqlite3.connect(path)
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        except Exception as exc:
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
            if dlg.exec() != QDialog.Accepted:
                return
            sim_id = dlg.selected_id()

        # Load (may take a moment for large databases)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            edges = load_edges(conn)
            if not edges:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self, "Error", "No edges found in database.")
                return

            bundle = load_road_data(conn, edges, sim_id)
            global_data = load_global_data(conn, sim_id)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{exc}")
            return
        finally:
            conn.close()

        if not bundle["densities"]:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self, "Error", f"No road_data found for simulation ID {sim_id}."
            )
            return

        self._initialize_app(edges, bundle, global_data)
        QApplication.restoreOverrideCursor()

    def _initialize_app(self, edges: list[dict], bundle: dict, global_data: list[dict]):
        self._edges = edges
        self._densities = bundle["densities"]
        self._obs_data = bundle["observables"]
        self._obs_domains = bundle["domains"]
        self._global_data = global_data
        self._current_idx = 0
        self._highlighted_edge = None

        # Precompute colour table (once; ~50–200 ms for typical datasets)
        self._precomputed = precompute_all_colors(self._obs_data, self._obs_domains)

        # Push edges to map
        self._map.set_edges(edges)
        self._map.highlighted_edge_id = None
        self._map.highlighted_node = None

        # Centre map on median geometry coordinate
        all_lats = [lat for e in edges for lon, lat in e["geometry"]]
        all_lons = [lon for e in edges for lon, lat in e["geometry"]]
        if all_lats:
            self._map.fit_bounds(
                min(all_lats),
                min(all_lons),
                max(all_lats),
                max(all_lons),
            )

        # Slider
        n = len(self._densities)
        self._slider.setMaximum(n - 1)
        self._slider.setValue(0)

        # Chart
        self._chart.set_global_data(global_data)

        # Legend
        self._refresh_legend()

        # First frame
        self._apply_timestep(0)
        self._set_data_loaded(True)
        self.statusBar().showMessage(f"Loaded {len(edges)} edges · {n} timesteps", 5000)

    # ── Visualization update ──────────────────────────────────────────────────

    def _apply_timestep(self, idx: int):
        """Push the pre-computed colour array and densities for timestep *idx*."""
        self._current_idx = idx

        key = self._selected_obs
        colors = self._precomputed.get(key, [[]])[idx] if self._precomputed else []
        dens = self._densities[idx]["densities"] if self._densities else []

        self._map.set_edge_colors(colors)
        self._map.set_edge_densities(dens)

        # Update time label
        dt = self._densities[idx]["datetime"]
        self._time_label.setText(dt.strftime("%Y-%m-%d %H:%M"))

        # Update chart marker
        self._chart.set_current_index(idx)

        # Refresh edge info if one is selected
        if self._highlighted_edge:
            self._show_edge_info(self._highlighted_edge, idx)

    def _refresh_legend(self):
        key = self._selected_obs
        domain = self._obs_domains.get(key, (0.0, 1.0))
        rev = EDGE_OBSERVABLE_CONFIG.get(key, {}).get("reverseColorScale", False)
        self._legend.set_observable(key, domain, rev)

    # ── Slider / playback ─────────────────────────────────────────────────────

    @Slot(int)
    def _on_slider_changed(self, value: int):
        if self._densities:
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
        n = len(self._densities)
        if n == 0:
            return
        next_idx = (self._current_idx + 1) % n
        self._slider.setValue(next_idx)

    @Slot(float)
    def _on_fps_changed(self, fps: float):
        if self._is_playing:
            self._timer.setInterval(max(1, int(1000 / fps)))

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
            self._refresh_legend()
            if self._densities:
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
        if self._densities and ts_idx < len(self._densities):
            dens_list = self._densities[ts_idx]["densities"]
            try:
                edge_pos = self._edges.index(edge)
                d = dens_list[edge_pos]
                density = f"{d:.2f}"
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
        self._play_btn.setEnabled(loaded)
        self._slider.setEnabled(loaded)
        self._obs_combo.setEnabled(loaded)
        self._inverse_btn.setEnabled(False)  # enabled per-selection
        self._clear_btn.setEnabled(loaded)
        self._edge_search_btn.setEnabled(loaded)
        self._node_search_btn.setEnabled(loaded)
        if not loaded:
            self._time_label.setText("—")

    def closeEvent(self, event):
        # Stop playback and tile loader threads cleanly
        self._timer.stop()
        self._map._pool.waitForDone(2000)
        super().closeEvent(event)
