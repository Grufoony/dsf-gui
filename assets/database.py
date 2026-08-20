"""
database.py
───────────
All SQLite querying and data-transformation logic, plus colour computation
that replaces the D3 colour-scale calls in the JS original.

Memory strategy
----------------
Time-series observables (density, speed, traveltime, n_observations,
queue_length) are stored as 2-D numpy float32 arrays shaped
(n_timesteps, n_edges) instead of nested Python lists of dicts/floats.

`edge_colors_for_timestep()` builds just the ~n_edges QColors needed for
the frame currently on screen, in O(n_edges) using vectorised numpy math.
That list is thrown away (garbage collected) the moment the next frame is
shown, so peak colour memory is O(edges), not O(edges * timesteps).
"""

from __future__ import annotations

import datetime
import sqlite3
from typing import Any

import numpy as np
from PySide6.QtGui import QColor

# ── Observable metadata ───────────────────────────────────────────────────────

EDGE_OBSERVABLE_CONFIG: dict[str, dict] = {
    "density": {"label": "Density", "reverseColorScale": False},
    "speed": {"label": "Speed", "reverseColorScale": True},
    "n_observations": {"label": "Observations", "reverseColorScale": False},
    "traveltime": {"label": "Travel Time", "reverseColorScale": False},
    "queue_length": {"label": "Queue Length", "reverseColorScale": False},
}

MAX_DENSITY: float = 200.0


# ── Colour helpers ────────────────────────────────────────────────────────────


def value_to_qcolor(
    value: float, dmin: float, dmax: float, reversed_: bool = False, alpha: int = 176
) -> QColor:
    """
    Map *value* in [dmin, dmax] to a QColor on the green→yellow→red scale.
    Pass reversed_=True for red→yellow→green (used for speed).
    alpha=176 ≈ 0.69 x 255, matching the JS rgba(…, 0.69).

    Kept for single-value use (e.g. tooltips); bulk colouring should use
    edge_colors_for_timestep() instead, which is vectorised.
    """
    rng = dmax - dmin
    t = 0.0 if rng == 0 else (value - dmin) / rng
    t = max(0.0, min(1.0, t))
    if reversed_:
        t = 1.0 - t

    if t <= 0.5:
        s = t * 2  # 0 → 1 over the first half
        r = int(s * 255)  #   0 → 255
        g = int(128 + s * 127)  # 128 → 255
    else:
        s = (t - 0.5) * 2  # 0 → 1 over the second half
        r = 255
        g = int((1.0 - s) * 255)  # 255 → 0

    return QColor(r, g, 0, alpha)


def _row_to_colors(
    values: np.ndarray,
    dmin: float,
    dmax: float,
    reversed_: bool = False,
    alpha: int = 176,
) -> list[QColor]:
    """Vectorised version of value_to_qcolor for a whole 1-D array of values."""
    rng = dmax - dmin
    if rng == 0:
        t = np.zeros_like(values, dtype=np.float32)
    else:
        t = (values.astype(np.float32) - dmin) / rng
    t = np.clip(t, 0.0, 1.0)
    if reversed_:
        t = 1.0 - t

    r = np.empty_like(t)
    g = np.empty_like(t)
    lo = t <= 0.5
    hi = ~lo

    s1 = t[lo] * 2
    r[lo] = s1 * 255
    g[lo] = 128 + s1 * 127

    s2 = (t[hi] - 0.5) * 2
    r[hi] = 255
    g[hi] = (1.0 - s2) * 255

    r_list = r.astype(np.uint8).tolist()
    g_list = g.astype(np.uint8).tolist()
    return [QColor(ri, gi, 0, alpha) for ri, gi in zip(r_list, g_list)]


def edge_colors_for_timestep(
    key: str,
    idx: int,
    values: dict[str, np.ndarray],
    domains: dict[str, tuple[float, float]],
    alpha: int = 176,
) -> list[QColor]:
    """
    Build the QColor list for ONE timestep of ONE observable, on demand.
    Call this from the UI whenever the slider moves / the observable
    changes - it's cheap (O(n_edges)) and replaces precompute_all_colors().
    """
    arr = values.get(key)
    if arr is None or idx >= len(arr):
        return []
    dmin, dmax = domains.get(key, (0.0, 1.0))
    rev = EDGE_OBSERVABLE_CONFIG.get(key, {}).get("reverseColorScale", False)
    row = np.nan_to_num(arr[idx], nan=0.0)
    return _row_to_colors(row, dmin, dmax, rev, alpha)


# ── Geometry parsing ──────────────────────────────────────────────────────────


def parse_linestring(wkt: str) -> list[tuple[float, float]]:
    """
    Parse a WKT LINESTRING into a list of (lon, lat) float tuples.

    Accepts both:
        LINESTRING (x0 y0, x1 y1, …)
        LINESTRING(x0 y0, x1 y1, …)
    """
    if not wkt:
        return []
    s = wkt.strip()
    # Strip the LINESTRING keyword and outer parentheses
    start = s.index("(") + 1
    end = s.rindex(")")
    coords_str = s[start:end]
    pts = []
    for part in coords_str.split(","):
        tokens = part.strip().split()
        if len(tokens) >= 2:
            pts.append((float(tokens[0]), float(tokens[1])))
    return pts


# ── Loaders ───────────────────────────────────────────────────────────────────


def get_simulations(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute("SELECT id, name FROM simulation_info ORDER BY id")
    return [{"id": row[0], "name": row[1] or f"Simulation {row[0]}"} for row in cur]


def load_edges(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT id, source, target, length, maxspeed, name, nlanes, geometry, coilcode "
        "FROM edges"
    )
    cols = [d[0] for d in cur.description]
    edges = []
    for row in cur:
        e: dict[str, Any] = dict(zip(cols, row))
        e["geometry"] = parse_linestring(e.get("geometry") or "")
        e["maxspeed"] = float(e.get("maxspeed") or 0)
        e["nlanes"] = int(e.get("nlanes") or 1)
        e["length"] = float(e.get("length") or 0)
        e["name"] = e.get("name") or ""
        edges.append(e)
    return edges


def _travel_time_expr(conn: sqlite3.Connection) -> str:
    """Return the SQL expression for travel time, adapting to the schema."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(road_data)")}
    if "traveltime" in cols:
        return "r.traveltime"
    if "travel_time" in cols:
        return "r.travel_time"
    if "travel_time_s" in cols:
        return "r.travel_time_s"
    return (
        "CASE WHEN r.avg_speed_kph > 0 "
        "THEN e.length / (r.avg_speed_kph / 3.6) ELSE 0 END"
    )


def load_road_data(
    conn: sqlite3.Connection,
    edges: list[dict],
    sim_id: int,
) -> dict[str, Any]:
    """
    Load per-edge time-series data for simulation *sim_id*.

    Returns
    -------
    {
        "datetimes": [dt, …],                       # length T
        "values": {                                  # each array shaped (T, n_edges), float32
            "density":        np.ndarray,
            "speed":          np.ndarray,
            "traveltime":     np.ndarray,
            "n_observations": np.ndarray,
            "queue_length":   np.ndarray,
        },
        "domains": {"density": (min, max), "speed": …, …},
    }

    """
    edge_ids = [e["id"] for e in edges]
    id_to_idx = {eid: i for i, eid in enumerate(edge_ids)}
    n_edges = len(edge_ids)
    tt_expr = _travel_time_expr(conn)

    cur = conn.execute(
        f"""
        SELECT r.datetime,
               r.street_id,
               r.density_vpk,
               r.avg_speed_kph,
               {tt_expr} AS tt,
               r.n_observations,
               r.queue_length
        FROM road_data r
        LEFT JOIN edges e ON e.id = r.street_id
        WHERE r.simulation_id = ?
        ORDER BY r.datetime, r.street_id
        """,
        (sim_id,),
    )

    datetimes: list[datetime.datetime] = []
    density_frames: list[np.ndarray] = []
    speed_frames: list[np.ndarray] = []
    tt_frames: list[np.ndarray] = []
    nobs_frames: list[np.ndarray] = []
    queue_frames: list[np.ndarray] = []

    cur_ts: Any = None
    dm = np.zeros(n_edges, dtype=np.float32)
    sm = np.zeros(n_edges, dtype=np.float32)
    tm = np.zeros(n_edges, dtype=np.float32)
    nm = np.zeros(n_edges, dtype=np.float32)
    qm = np.zeros(n_edges, dtype=np.float32)

    def _flush(ts_str: str):
        datetimes.append(_parse_dt(ts_str))
        # .copy() so the next frame's in-place writes don't corrupt this one
        density_frames.append(dm.copy())
        speed_frames.append(sm.copy())
        tt_frames.append(tm.copy())
        nobs_frames.append(nm.copy())
        queue_frames.append(qm.copy())

    for ts, sid, d, s, t, n, q in cur:
        if ts != cur_ts:
            if cur_ts is not None:
                _flush(cur_ts)
            cur_ts = ts
            dm.fill(0)
            sm.fill(0)
            tm.fill(0)
            nm.fill(0)
            qm.fill(0)
        i = id_to_idx.get(sid)
        if i is None:
            continue  # road_data row references an edge we don't have
        dm[i] = d or 0
        sm[i] = s or 0
        tm[i] = t or 0
        nm[i] = n or 0
        qm[i] = q or 0

    if cur_ts is not None:
        _flush(cur_ts)

    def _stack(frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros((0, n_edges), dtype=np.float32)
        return np.stack(frames)

    values: dict[str, np.ndarray] = {
        "density": _stack(density_frames),
        "speed": _stack(speed_frames),
        "traveltime": _stack(tt_frames),
        "n_observations": _stack(nobs_frames),
        "queue_length": _stack(queue_frames),
    }

    def _domain(arr: np.ndarray) -> tuple[float, float]:
        if arr.size == 0:
            return (0.0, 1.0)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return (0.0, 1.0)
        mn, mx = float(finite.min()), float(finite.max())
        return (mn, mn + 1.0) if mn == mx else (mn, mx)

    domains = {k: _domain(v) for k, v in values.items()}

    return {"datetimes": datetimes, "values": values, "domains": domains}


def load_global_data(
    conn: sqlite3.Connection,
    sim_id: int,
) -> list[dict[str, Any]]:
    """
    Load aggregated per-timestep statistics (for the chart).
    Tries avg_stats / avgstats first, falls back to GROUP BY on road_data.
    """
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }

    agg_table = (
        "avg_stats"
        if "avg_stats" in tables
        else "avgstats"
        if "avgstats" in tables
        else None
    )

    if agg_table:
        skip = {"id", "simulation_id", "datetime", "time_step"}
        metric_cols = [
            row[1]
            for row in conn.execute(f"PRAGMA table_info({agg_table})")
            if row[1] not in skip
        ]
        if metric_cols:
            col_sql = ", ".join(metric_cols)
            cur = conn.execute(
                f"SELECT datetime, {col_sql} "
                f"FROM {agg_table} "
                f"WHERE simulation_id = ? ORDER BY datetime",
                (sim_id,),
            )
            rows = []
            for r in cur:
                entry: dict[str, Any] = {"datetime": _parse_dt(r[0])}
                for i, col in enumerate(metric_cols):
                    entry[col] = float(r[i + 1] or 0)
                rows.append(entry)
            return rows

    road_cols = {row[1] for row in conn.execute("PRAGMA table_info(road_data)")}
    count_expr = "SUM(counts)" if "counts" in road_cols else "COUNT(*)"
    cur = conn.execute(
        f"""
        SELECT datetime,
               AVG(density_vpk)  AS mean_density_vpk,
               AVG(avg_speed_kph) AS mean_speed_kph,
               {count_expr}       AS total_count
        FROM road_data
        WHERE simulation_id = ?
        GROUP BY datetime
        ORDER BY datetime
        """,
        (sim_id,),
    )
    rows = []
    for r in cur:
        rows.append(
            {
                "datetime": _parse_dt(r[0]),
                "mean_density_vpk": float(r[1] or 0),
                "mean_speed_kph": float(r[2] or 0),
                "total_count": float(r[3] or 0),
            }
        )
    return rows


# ── Internal ──────────────────────────────────────────────────────────────────


def _parse_dt(value: Any) -> datetime.datetime:
    """Parse a datetime value that may be a string or already a datetime."""
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.fromisoformat(str(value))
