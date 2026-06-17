"""
database.py
───────────
All SQLite querying and data-transformation logic, plus the colour
precomputation that replaces the D3 colour-scale calls in the JS original.
"""

from __future__ import annotations

import datetime
import math
import sqlite3
from typing import Any

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


def precompute_all_colors(
    obs_data: dict[str, list[dict]],
    obs_domains: dict[str, tuple[float, float]],
) -> dict[str, list[list[QColor]]]:
    """
    Build a complete colour lookup table so that slider updates are O(1).

    Returns
    -------
    {observable_key: [[QColor per edge] per timestep]}
    """
    result: dict[str, list[list[QColor]]] = {}
    for key, rows in obs_data.items():
        dmin, dmax = obs_domains.get(key, (0.0, 1.0))
        rev = EDGE_OBSERVABLE_CONFIG.get(key, {}).get("reverseColorScale", False)
        result[key] = [
            [
                value_to_qcolor(
                    v if (v is not None and not math.isnan(float(v))) else 0.0,
                    dmin,
                    dmax,
                    rev,
                )
                for v in row["values"]
            ]
            for row in rows
        ]
    return result


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
    # Fallback: derive from length and speed
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
        "densities":   [{"datetime": dt, "densities": [float, …]}, …],
        "observables": {
            "density":      [{"datetime": dt, "values": [float, …]}, …],
            "speed":        …,
            "n_observations": …,
            "traveltime":   …,
            "queue_length": …,
        },
        "domains": {"density": (min, max), "speed": …, …},
    }
    """
    edge_ids = [e["id"] for e in edges]
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

    # Accumulate rows grouped by timestamp
    density_rows: list[dict] = []
    speed_rows: list[dict] = []
    traveltime_rows: list[dict] = []
    n_observations_rows: list[dict] = []
    queuelength_rows: list[dict] = []

    cur_ts: Any = None
    dm: dict = {}
    sm: dict = {}
    tm: dict = {}
    nm: dict = {}
    qm: dict = {}
    def _flush(ts_str: str):
        dt = _parse_dt(ts_str)
        dn = [float(dm.get(eid) or 0) for eid in edge_ids]
        sn = [float(sm.get(eid) or 0) for eid in edge_ids]
        tn = [float(tm.get(eid) or 0) for eid in edge_ids]
        qn = [float(qm.get(eid) or 0) for eid in edge_ids]
        density_rows.append({"datetime": dt, "densities": dn})
        speed_rows.append({"datetime": dt, "values": sn})
        traveltime_rows.append({"datetime": dt, "values": tn})
        n_observations_rows.append({"datetime": dt, "values": [float(nm.get(eid) or 0) for eid in edge_ids]})
        queuelength_rows.append({"datetime": dt, "values": qn})

    for ts, sid, d, s, t, n, q in cur:
        if ts != cur_ts:
            if cur_ts is not None:
                _flush(cur_ts)
            cur_ts = ts
            dm = {}
            sm = {}
            tm = {}
            nm = {}
            qm = {}
        dm[sid] = d
        sm[sid] = s
        tm[sid] = t
        nm[sid] = n
        qm[sid] = q

    if cur_ts is not None:
        _flush(cur_ts)

    obs: dict[str, list[dict]] = {
        "density": [
            {"datetime": r["datetime"], "values": r["densities"]} for r in density_rows
        ],
        "speed": speed_rows,
        "traveltime": traveltime_rows,
        "queue_length": queuelength_rows,
        "n_observations": n_observations_rows,
    }

    def _domain(rows: list[dict]) -> tuple[float, float]:
        vals = [
            v
            for r in rows
            for v in r["values"]
            if v is not None and math.isfinite(float(v))
        ]
        if not vals:
            return (0.0, 1.0)
        mn, mx = min(vals), max(vals)
        return (mn, mn + 1.0) if mn == mx else (mn, mx)

    domains = {k: _domain(v) for k, v in obs.items()}

    return {"densities": density_rows, "observables": obs, "domains": domains}


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

    # Fallback: compute averages on-the-fly
    # (counts column may not exist; use a safe expression)
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
