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
into world-pixel coordinates per zoom level and cache the result, along with
a ready-to-draw QPolygonF, a bounding box (for viewport culling), and a
spatial grid index (for O(1)-ish hit testing).  Because these are all in
*world*-pixel space, panning never touches them: the QPainter is translated
once per frame and the cached polygons are drawn as-is — no per-point
Python-level coordinate math on every repaint.

Performance notes
──────────────────
- Edge polygons are built once per zoom level (see _ensure_geo_cache) and
  drawn via a single QPainter.translate(), instead of rebuilding a
  QPolygonF from scratch (with a Python list comprehension) on every paint.
- Edges whose bounding box doesn't intersect the viewport are skipped
  entirely (viewport culling).
- Placeholder tiles (scaled-up parent tiles shown while the real tile is
  still loading) are cached per (zoom, tx, ty, filter) so the crop/scale/
  filter work isn't repeated on every repaint during a pan.
- Hit testing (hover + click) uses a coarse spatial grid so we only test
  segments of edges near the cursor instead of every edge in the dataset.
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

    # Spatial-grid cell size (world pixels) used to bucket edges for hit testing.
    _GRID_CELL = 512

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
        # Ready-to-draw polygons in world-pixel space, parallel to _geo_cache.
        self._geo_polygons: list[QPolygonF] = []
        # (minx, miny, maxx, maxy) per edge in world-pixel space, or None.
        self._geo_bounds: list[Optional[tuple[float, float, float, float]]] = []
        # Spatial grid for hit testing: {(gx, gy): [edge_index, ...]}
        self._grid: dict[tuple[int, int], list[int]] = {}
        self._geo_zoom: Optional[int] = None

        # ── Tile cache ─────────────────────────────────────────────────────
        self._raw: dict[tuple, QPixmap] = {}  # {(z, tx, ty): raw pixmap}
        self._filt: dict[tuple, QPixmap] = {}  # {(z, tx, ty): filtered pixmap}
        # {(z, tx, ty, filter): scaled placeholder pixmap} — avoids redoing
        # the crop/scale/filter work on every repaint while a tile loads.
        self._placeholder_cache: dict[tuple, QPixmap] = {}
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
        self._placeholder_cache.clear()
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
        """
        Project edge geometries to world pixels and build the draw-ready
        QPolygonF list, bounding boxes, and spatial grid.  No-op if zoom
        hasn't changed since the last build.
        """
        if self._geo_zoom == self.zoom and self._geo_cache:
            return

        cell = self._GRID_CELL
        self._geo_cache = []
        self._geo_polygons = []
        self._geo_bounds = []
        self._grid = {}

        for idx, edge in enumerate(self.edges):
            pts = [lat_lon_to_world(lat, lon, self.zoom) for lon, lat in edge["geometry"]]
            self._geo_cache.append(pts)

            if len(pts) >= 2:
                self._geo_polygons.append(QPolygonF([QPointF(x, y) for x, y in pts]))
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bounds = (min(xs), min(ys), max(xs), max(ys))
                self._geo_bounds.append(bounds)

                gx0, gy0 = int(bounds[0] // cell), int(bounds[1] // cell)
                gx1, gy1 = int(bounds[2] // cell), int(bounds[3] // cell)
                for gx in range(gx0, gx1 + 1):
                    for gy in range(gy0, gy1 + 1):
                        self._grid.setdefault((gx, gy), []).append(idx)
            else:
                self._geo_polygons.append(QPolygonF())
                self._geo_bounds.append(None)

        self._geo_zoom = self.zoom

    def _candidate_edges(self, wx: float, wy: float, thr: float) -> set[int]:
        """Return indices of edges whose grid cells are near (wx, wy)."""
        cell = self._GRID_CELL
        gx0, gy0 = int((wx - thr) // cell), int((wy - thr) // cell)
        gx1, gy1 = int((wx + thr) // cell), int((wy + thr) // cell)
        candidates: set[int] = set()
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                bucket = self._grid.get((gx, gy))
                if bucket:
                    candidates.update(bucket)
        return candidates

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
            # This tile may now serve as a source for placeholders at other
            # zoom levels (or be drawn directly itself); either way any
            # cached placeholder crops that predate it are stale. Tile loads
            # are infrequent relative to repaints, so this is cheap.
            self._placeholder_cache.clear()
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
        """Return a scaled parent tile while the real one loads (cached)."""
        cache_key = (self.zoom, tx, ty, self.tile_filter)
        cached = self._placeholder_cache.get(cache_key)
        if cached is not None:
            return cached

        for dz in range(1, 5):
            pz = self.zoom - dz
            if pz < 0:
                break
            ptx, pty = tx >> dz, ty >> dz
            # Reuse the already-cached filtered parent tile instead of
            # re-running _apply_filter on every call.
            src = self._filtered_tile(pz, ptx, pty)
            if src is None:
                continue
            portion = 256 >> dz
            ox = (tx % (1 << dz)) * portion
            oy = (ty % (1 << dz)) * portion
            cropped = src.copy(ox, oy, portion, portion)
            result = cropped.scaled(
                256, 256, Qt.IgnoreAspectRatio, Qt.FastTransformation
            )
            self._placeholder_cache[cache_key] = result
            return result
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

        # Visible world-pixel rect, for cheap bounding-box culling.
        vx0, vy0 = cx - hw, cy - hh
        vx1, vy1 = cx + hw, cy + hh

        pen = QPen()
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)

        hi_id = self.highlighted_edge_id

        # Draw everything in world-pixel space; a single translate maps it
        # onto the screen, so cached polygons need no per-point rebuilding.
        p.save()
        p.translate(-cx + hw, -cy + hh)

        for i, (edge, poly, bounds) in enumerate(
            zip(self.edges, self._geo_polygons, self._geo_bounds)
        ):
            if bounds is None:
                continue
            bminx, bminy, bmaxx, bmaxy = bounds
            if bmaxx < vx0 or bminx > vx1 or bmaxy < vy0 or bminy > vy1:
                continue  # outside viewport — skip entirely
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
                poly = self._geo_polygons[i]
                if poly.isEmpty():
                    break
                density = (
                    self.edge_densities[i] if i < len(self.edge_densities) else 0.0
                )
                df = min(density / MAX_D, 2.0)
                width = max(1.0, base_w * (0.5 + df)) * 1.5
                pen.setColor(QColor(255, 255, 255, 230))
                pen.setWidthF(width)
                p.setPen(pen)
                p.drawPolyline(poly)
                break

        p.restore()

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
        self._placeholder_cache.clear()
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
        if not self.edges:
            return False
        self._ensure_geo_cache()
        wx, wy = self._s2w(sx, sy)
        for i in self._candidate_edges(wx, wy, thr):
            pts = self._geo_cache[i]
            for j in range(len(pts) - 1):
                ax, ay = pts[j]
                bx, by = pts[j + 1]
                if self._seg_dist(wx, wy, ax, ay, bx, by) < thr:
                    return True
        return False

    def _handle_click(self, sx: float, sy: float, thr: float = 10.0):
        if not self.edges:
            return
        self._ensure_geo_cache()
        wx, wy = self._s2w(sx, sy)
        best_d, best_i = float("inf"), -1
        for i in self._candidate_edges(wx, wy, thr):
            pts = self._geo_cache[i]
            for j in range(len(pts) - 1):
                ax, ay = pts[j]
                bx, by = pts[j + 1]
                d = self._seg_dist(wx, wy, ax, ay, bx, by)
                if d < best_d:
                    best_d, best_i = d, i
        if best_i >= 0 and best_d < thr:
            self.highlighted_edge_id = self.edges[best_i]["id"]
            self.edge_clicked.emit(self.edges[best_i], best_i)
            self.update()