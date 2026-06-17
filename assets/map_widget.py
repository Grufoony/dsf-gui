"""
map_widget.py
─────────────
Custom PySide6 widget that combines an OpenStreetMap slippy-map tile layer
with a QPainter-based road-network edge overlay.

Coordinate systems
──────────────────
World pixels (wx, wy)
    Mercator pixel coordinates at the current zoom level.
    tile (tx, ty) spans world pixels [tx*256, (tx+1)*256) × [ty*256, (ty+1)*256).
    Stable across pans; only change on zoom.

Screen pixels (sx, sy)
    Relative to this widget's top-left corner.
    sx = wx − cx + width()/2      (cx = world-pixel x of map centre)
    sy = wy − cy + height()/2

Geometry cache
──────────────
Edge geometries are stored as lat/lon WKT in the DB.  We project them once
into world-pixel coordinates per zoom level and cache the result.  On pan,
converting to screen coords is just two subtractions—no trig.
"""

from __future__ import annotations

import math
from typing import Optional
import requests

from PySide6.QtCore import (
    Qt,
    QObject,
    QPoint,
    QPointF,
    QRunnable,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget


# ── Coordinate helpers ────────────────────────────────────────────────────────


def lat_lon_to_world(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Latitude / longitude → world-pixel coordinates at *zoom*."""
    n = 256 << zoom  # 256 × 2**zoom
    x = (lon + 180.0) / 360.0 * n
    lr = math.radians(lat)
    y = (1.0 - math.log(math.tan(lr) + 1.0 / math.cos(lr)) / math.pi) / 2.0 * n
    return x, y


def world_to_lat_lon(wx: float, wy: float, zoom: int) -> tuple[float, float]:
    """World-pixel coordinates → latitude / longitude."""
    n = 256 << zoom
    lon = wx / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * wy / n))))
    return lat, lon


# ── Tile filter ───────────────────────────────────────────────────────────────


def _apply_filter(pixmap: QPixmap, mode: str) -> QPixmap:
    """Return a new QPixmap with 'normal', 'gray', or 'invert' applied."""
    if mode == "normal":
        return pixmap

    img = pixmap.toImage().convertToFormat(QImage.Format_ARGB32)

    if mode in ("gray", "invert"):
        img = img.convertToFormat(QImage.Format_Grayscale8).convertToFormat(
            QImage.Format_ARGB32
        )

    if mode == "invert":
        # CompositionMode_Difference with white = bitwise invert of each channel
        p = QPainter(img)
        p.setCompositionMode(QPainter.CompositionMode_Difference)
        p.fillRect(img.rect(), QColor(255, 255, 255))
        p.end()

    return QPixmap.fromImage(img)


# ── Async tile loader ─────────────────────────────────────────────────────────
#
# IMPORTANT – Qt threading rule:
#   QPixmap can only be created / manipulated in the main GUI thread.
#   Worker threads must NOT touch QPixmap.
#
# Fix: the worker emits the raw PNG bytes; _on_tile_ready (main thread)
#      converts them to QPixmap safely.


class _TileSig(QObject):
    # Carries raw PNG bytes so the worker never touches QPixmap.
    ready = Signal(int, int, int, bytes)  # zoom, tx, ty, png_bytes


class _TileJob(QRunnable):
    def __init__(self, zoom: int, tx: int, ty: int, sig: _TileSig):
        super().__init__()
        self._z, self._tx, self._ty = zoom, tx, ty
        self._sig = sig
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        sd = ("a", "b", "c")[(self._tx + self._ty) % 3]
        url = f"https://{sd}.tile.openstreetmap.org/{self._z}/{self._tx}/{self._ty}.png"
        try:
            r = requests.get(
                url,
                headers={"User-Agent": "RoadVizPy/1.0"},
                timeout=15,
            )
            if r.status_code == 200 and r.content:
                # Emit raw bytes — QPixmap is created in the main thread below.
                self._sig.ready.emit(self._z, self._tx, self._ty, r.content)
        except Exception:
            pass


# ── Map widget ────────────────────────────────────────────────────────────────


class MapWidget(QWidget):
    """
    Slippy map + road-network edge overlay.

    Public signals
    ──────────────
    edge_clicked(edge: dict, index: int)
        Emitted when the user clicks near a road edge.
    """

    edge_clicked = Signal(dict, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # ── Map state ──────────────────────────────────────────────────────
        self.zoom: int = 13
        self._cx: float = 0.0  # world-pixel x of viewport centre
        self._cy: float = 0.0  # world-pixel y of viewport centre

        # ── Edge data ──────────────────────────────────────────────────────
        self.edges: list[dict] = []
        self.edge_colors: list[QColor] = []
        self.edge_densities: list[float] = []
        self.highlighted_edge_id = None
        self.highlighted_node: Optional[tuple[float, float]] = None  # (lon, lat)

        # Geometry cache: [[(wx, wy), …], …] valid for self._geo_zoom
        self._geo_cache: list[list[tuple[float, float]]] = []
        self._geo_zoom: Optional[int] = None

        # ── Tile cache ─────────────────────────────────────────────────────
        self._raw: dict[tuple, QPixmap] = {}  # {(z, tx, ty): raw pixmap}
        self._filt: dict[tuple, QPixmap] = {}  # {(z, tx, ty): filtered pixmap}
        self._pending: set[tuple] = set()
        self.tile_filter: str = "invert"  # 'normal' | 'gray' | 'invert'

        # ── Tile thread pool ───────────────────────────────────────────────
        self._sig = _TileSig()
        self._sig.ready.connect(self._on_tile_ready)
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(8)

        # ── Drag state ─────────────────────────────────────────────────────
        self._drag_start: Optional[QPoint] = None
        self._drag_cx: float = 0.0
        self._drag_cy: float = 0.0

        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.StrongFocus)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_center(self, lat: float, lon: float, zoom: Optional[int] = None):
        """Pan (and optionally zoom) to the given coordinate."""
        if zoom is not None:
            self.zoom = max(0, min(19, zoom))
        self._cx, self._cy = lat_lon_to_world(lat, lon, self.zoom)
        self._geo_zoom = None
        self.update()
        self._request_tiles()

    def set_edges(self, edges: list[dict]):
        self.edges = edges
        self._geo_zoom = None
        n = len(edges)
        self.edge_colors = [QColor(0, 128, 0, 176)] * n
        self.edge_densities = [0.0] * n
        self.update()

    def set_edge_colors(self, colors: list[QColor]):
        self.edge_colors = colors
        self.update()

    def set_edge_densities(self, densities: list[float]):
        self.edge_densities = densities
        self.update()

    def set_tile_filter(self, mode: str):
        self.tile_filter = mode
        self._filt.clear()  # regenerated from raw on next paint
        self.update()

    def fit_bounds(
        self,
        lat_min: float,
        lon_min: float,
        lat_max: float,
        lon_max: float,
        padding: int = 50,
    ):
        """Zoom/pan so the given bounding box fits the viewport."""
        clat = (lat_min + lat_max) / 2
        clon = (lon_min + lon_max) / 2
        for z in range(18, 0, -1):
            wx1, wy1 = lat_lon_to_world(lat_min, lon_min, z)
            wx2, wy2 = lat_lon_to_world(lat_max, lon_max, z)
            if (
                abs(wx2 - wx1) + 2 * padding < self.width()
                and abs(wy1 - wy2) + 2 * padding < self.height()
            ):
                self.zoom = z
                break
        self.set_center(clat, clon)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _w2s(self, wx: float, wy: float) -> tuple[float, float]:
        """World pixel → screen pixel."""
        return wx - self._cx + self.width() * 0.5, wy - self._cy + self.height() * 0.5

    def _s2w(self, sx: float, sy: float) -> tuple[float, float]:
        """Screen pixel → world pixel."""
        return sx + self._cx - self.width() * 0.5, sy + self._cy - self.height() * 0.5

    def _ensure_geo_cache(self):
        """Project edge geometries to world pixels (no-op if zoom unchanged)."""
        if self._geo_zoom == self.zoom and self._geo_cache:
            return
        self._geo_cache = [
            [lat_lon_to_world(lat, lon, self.zoom) for lon, lat in edge["geometry"]]
            for edge in self.edges
        ]
        self._geo_zoom = self.zoom

    def _visible_tile_range(self) -> tuple[int, int, int, int]:
        w, h = self.width(), self.height()
        left, top = self._cx - w * 0.5, self._cy - h * 0.5
        mt = (1 << self.zoom) - 1
        tx0 = max(0, int(left / 256))
        ty0 = max(0, int(top / 256))
        tx1 = min(mt, int((left + w) / 256))
        ty1 = min(mt, int((top + h) / 256))
        return tx0, ty0, tx1, ty1

    def _request_tiles(self):
        tx0, ty0, tx1, ty1 = self._visible_tile_range()
        z = self.zoom
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                key = (z, tx, ty)
                if key not in self._raw and key not in self._pending:
                    self._pending.add(key)
                    self._pool.start(_TileJob(z, tx, ty, self._sig))

    @Slot(int, int, int, bytes)
    def _on_tile_ready(self, z: int, tx: int, ty: int, data: bytes):
        """
        Called in the main thread via a queued signal connection.
        QPixmap is created here — the only safe place.
        """
        key = (z, tx, ty)
        self._pending.discard(key)
        pm = QPixmap()
        if pm.loadFromData(data):
            self._raw[key] = pm
            self.update()

    def _filtered_tile(self, z: int, tx: int, ty: int) -> Optional[QPixmap]:
        key = (z, tx, ty)
        raw = self._raw.get(key)
        if raw is None:
            return None
        if key not in self._filt:
            self._filt[key] = _apply_filter(raw, self.tile_filter)
        return self._filt[key]

    def _placeholder_tile(self, tx: int, ty: int) -> Optional[QPixmap]:
        """Return a scaled parent tile while the real one loads."""
        for dz in range(1, 5):
            pz = self.zoom - dz
            if pz < 0:
                break
            ptx, pty = tx >> dz, ty >> dz
            key = (pz, ptx, pty)
            if key in self._raw:
                portion = 256 >> dz
                ox = (tx % (1 << dz)) * portion
                oy = (ty % (1 << dz)) * portion
                src = _apply_filter(self._raw[key], self.tile_filter)
                cropped = src.copy(ox, oy, portion, portion)
                return cropped.scaled(
                    256, 256, Qt.IgnoreAspectRatio, Qt.FastTransformation
                )
        return None

    # ── Painting ──────────────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 30, 30))

        self._draw_tiles(p)

        if self.edges:
            self._ensure_geo_cache()
            self._draw_edges(p)

        if self.highlighted_node:
            self._draw_node_highlight(p)

        p.end()

    def _draw_tiles(self, p: QPainter):
        left = self._cx - self.width() * 0.5
        top = self._cy - self.height() * 0.5
        tx0, ty0, tx1, ty1 = self._visible_tile_range()
        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                pm = self._filtered_tile(self.zoom, tx, ty) or self._placeholder_tile(
                    tx, ty
                )
                if pm:
                    p.drawPixmap(int(tx * 256 - left), int(ty * 256 - top), pm)

    def _draw_edges(self, p: QPainter):
        MAX_D = 200.0
        base_w = max(1.0, 3.0 + self.zoom - 13)
        cx, cy = self._cx, self._cy
        hw, hh = self.width() * 0.5, self.height() * 0.5

        pen = QPen()
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)

        hi_id = self.highlighted_edge_id

        for i, (edge, pts) in enumerate(zip(self.edges, self._geo_cache)):
            if len(pts) < 2:
                continue
            if hi_id and edge["id"] == hi_id:
                continue  # drawn last, on top

            density = self.edge_densities[i] if i < len(self.edge_densities) else 0.0
            df = min(density / MAX_D, 2.0)
            width = max(1.0, base_w * (0.5 + df))
            color = (
                self.edge_colors[i]
                if i < len(self.edge_colors)
                else QColor(0, 128, 0, 176)
            )

            pen.setColor(color)
            pen.setWidthF(width)
            p.setPen(pen)

            poly = QPolygonF([QPointF(wx - cx + hw, wy - cy + hh) for wx, wy in pts])
            p.drawPolyline(poly)

            # Autostrada dashed overlay
            name = edge.get("name", "") or ""
            if "autostrada" in name.lower():
                dash = QPen(color, width, Qt.CustomDashLine)
                dash.setDashPattern([3.0, 3.0])
                dash.setCapStyle(Qt.RoundCap)
                p.setPen(dash)
                p.drawPolyline(poly)
                p.setPen(pen)

        # Draw highlighted edge last (always on top, white)
        if hi_id:
            for i, edge in enumerate(self.edges):
                if edge["id"] != hi_id:
                    continue
                pts = self._geo_cache[i]
                if len(pts) < 2:
                    break
                density = (
                    self.edge_densities[i] if i < len(self.edge_densities) else 0.0
                )
                df = min(density / MAX_D, 2.0)
                width = max(1.0, base_w * (0.5 + df)) * 1.5
                pen.setColor(QColor(255, 255, 255, 230))
                pen.setWidthF(width)
                p.setPen(pen)
                poly = QPolygonF(
                    [QPointF(wx - cx + hw, wy - cy + hh) for wx, wy in pts]
                )
                p.drawPolyline(poly)
                break

    def _draw_node_highlight(self, p: QPainter):
        lon, lat = self.highlighted_node  # type: ignore[misc]
        wx, wy = lat_lon_to_world(lat, lon, self.zoom)
        sx, sy = self._w2s(wx, wy)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.setBrush(QColor(255, 255, 255, 180))
        p.drawEllipse(QPointF(sx, sy), 10, 10)

    # ── Mouse / wheel ─────────────────────────────────────────────────────────

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag_start = e.position().toPoint()
            self._drag_cx = self._cx
            self._drag_cy = self._cy
            self.setCursor(QCursor(Qt.ClosedHandCursor))

    def mouseMoveEvent(self, e):
        if self._drag_start is not None:
            d = e.position().toPoint() - self._drag_start
            self._cx = self._drag_cx - d.x()
            self._cy = self._drag_cy - d.y()
            self.update()
        else:
            near = self._is_near_edge(e.position().x(), e.position().y())
            self.setCursor(QCursor(Qt.PointingHandCursor if near else Qt.ArrowCursor))

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._drag_start is not None:
            dist = (e.position().toPoint() - self._drag_start).manhattanLength()
            self._drag_start = None
            self.setCursor(QCursor(Qt.ArrowCursor))
            if dist < 5:
                self._handle_click(e.position().x(), e.position().y())
            else:
                self._request_tiles()

    def wheelEvent(self, e):
        dy = e.angleDelta().y()
        if dy == 0:
            return
        new_z = min(19, self.zoom + 1) if dy > 0 else max(0, self.zoom - 1)
        if new_z == self.zoom:
            return

        # Keep the world point under the cursor stationary
        mx, my = e.position().x(), e.position().y()
        bwx, bwy = self._s2w(mx, my)
        scale = 2 ** (new_z - self.zoom)

        self.zoom = new_z
        # new_center = scaled_mouse_world − mouse_screen_offset
        self._cx = bwx * scale - mx + self.width() * 0.5
        self._cy = bwy * scale - my + self.height() * 0.5

        self._geo_zoom = None
        self._filt.clear()
        self.update()
        self._request_tiles()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._request_tiles()

    def showEvent(self, e):
        super().showEvent(e)
        self._request_tiles()

    # ── Hit testing ───────────────────────────────────────────────────────────

    @staticmethod
    def _seg_dist(px, py, ax, ay, bx, by) -> float:
        dx, dy = bx - ax, by - ay
        lsq = dx * dx + dy * dy
        if lsq == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / lsq))
        return math.hypot(px - ax - t * dx, py - ay - t * dy)

    def _is_near_edge(self, sx: float, sy: float, thr: float = 8.0) -> bool:
        if not self._geo_cache:
            return False
        cx, cy = self._cx, self._cy
        hw, hh = self.width() * 0.5, self.height() * 0.5
        for pts in self._geo_cache:
            for i in range(len(pts) - 1):
                ax = pts[i][0] - cx + hw
                ay = pts[i][1] - cy + hh
                bx = pts[i + 1][0] - cx + hw
                by = pts[i + 1][1] - cy + hh
                if self._seg_dist(sx, sy, ax, ay, bx, by) < thr:
                    return True
        return False

    def _handle_click(self, sx: float, sy: float, thr: float = 10.0):
        if not self._geo_cache:
            return
        self._ensure_geo_cache()
        cx, cy = self._cx, self._cy
        hw, hh = self.width() * 0.5, self.height() * 0.5
        best_d, best_i = float("inf"), -1
        for i, pts in enumerate(self._geo_cache):
            for j in range(len(pts) - 1):
                ax = pts[j][0] - cx + hw
                ay = pts[j][1] - cy + hh
                bx = pts[j + 1][0] - cx + hw
                by = pts[j + 1][1] - cy + hh
                d = self._seg_dist(sx, sy, ax, ay, bx, by)
                if d < best_d:
                    best_d, best_i = d, i
        if best_i >= 0 and best_d < thr:
            self.highlighted_edge_id = self.edges[best_i]["id"]
            self.edge_clicked.emit(self.edges[best_i], best_i)
            self.update()
