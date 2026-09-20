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
from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtGui import QColor

# ── Observable metadata ───────────────────────────────────────────────────────

EDGE_OBSERVABLE_CONFIG: dict[str, dict] = {
    "density": {"label": "Density", "reverseColorScale": False},
    "density_norm": {"label": "Density (normalized)", "reverseColorScale": False},
    "speed": {"label": "Speed", "reverseColorScale": True},
    "n_observations": {"label": "Observations", "reverseColorScale": False},
    "traveltime": {"label": "Travel Time", "reverseColorScale": False},
    "queue_length": {"label": "Queue Length", "reverseColorScale": False},
}

# Observable selected on startup: every edge measured against its own capacity.
DEFAULT_OBSERVABLE: str = "density_norm"

# Occupancy at which the normalised-density scale reaches red. Yellow is the
# midpoint of the ramp, so it lands at half this value (0.15), and anything
# above stays red all the way to 1. Real traffic sits far below jam density,
# so a scale that only saturates at 1.0 leaves the map uniformly blue.
DENSITY_NORM_SATURATION: float = 0.3

# Mean vehicle length is not written to the database, but it is recoverable
# from it (see estimate_mean_vehicle_length). This is only the fallback for a
# run that never fills an edge: dsf::mobility::Road::m_meanVehicleLength.
DSF_DEFAULT_VEHICLE_LENGTH_M: float = 5.0

# A recovered length outside this range means the run never came close to
# saturating anything and the estimate is an artefact, not a measurement.
PLAUSIBLE_VEHICLE_LENGTH_M: tuple[float, float] = (2.0, 20.0)

# How often load_road_data reports progress. ~30 calls over a 3.6M-row
# simulation: often enough to feel live, rare enough to cost nothing.
PROGRESS_ROW_INTERVAL = 100_000


# ── Per-edge normalisation ────────────────────────────────────────────────────


def lane_metres(edges: list[dict[str, Any]]) -> np.ndarray:
    """``length_m * nlanes`` per edge — the quantity DSF divides agents by."""
    length = np.array([float(e.get("length") or 0.0) for e in edges], dtype=np.float64)
    nlanes = np.array(
        [max(1, int(e.get("nlanes") or 1)) for e in edges], dtype=np.float64
    )
    return length * nlanes


def implied_agents(density_vpk: np.ndarray, edges: list[dict[str, Any]]) -> np.ndarray:
    """
    Recover DSF's agent count from a stored density.

    DSF writes ``density_vpk = nAgents / (length_m * nlanes) * 1000`` (see
    FirstOrderDynamics.cpp: ``pStreet->density<false>() * 1e3``) — vehicles per
    *lane*-kilometre, not per kilometre of road. Multiplying back by the lane
    metres returns the agent count exactly, give or take the half-agents DSF
    counts while a vehicle is between two streets.
    """
    return density_vpk * lane_metres(edges) / 1000.0


def estimate_mean_vehicle_length(
    conn: sqlite3.Connection, edges: list[dict[str, Any]]
) -> tuple[float, int]:
    """
    Recover the mean vehicle length the simulation ran with, from the data.

    DSF stores neither the length nor the capacity (TrafficSimulator.cpp writes
    only id/source/target/length/maxspeed/name/nlanes/coilcode/geometry), but it
    enforces ``nAgents <= capacity = ceil(length_m * nlanes / L)`` on every
    street at every step. Each edge that was ever busy therefore bounds L:

        ceil(x / L) >= n   =>   x / L > n - 1   =>   L < x / (n - 1)

    for ``x`` lane metres and ``n`` the most agents ever seen on it. The
    tightest of those bounds is the estimate — edges that actually filled up
    drive it, and the more of them there are the closer it is pinned.

    The peak is taken over *every* simulation in the file, not just the one
    being displayed: the length is a property of the network, one quiet run
    would leave it barely constrained, and a bound from any run is valid for
    all of them.

    Returns (length_m, n_witnesses), where the witnesses are the edges whose
    bound lands within 1% of the estimate; one lone witness means a single busy
    edge is carrying the whole inference. Returns (nan, 0) when nothing was ever
    busy enough to constrain anything.
    """
    id_to_idx = {e["id"]: i for i, e in enumerate(edges)}
    peak = np.zeros(len(edges), dtype=np.float64)
    for street_id, max_density in conn.execute(
        "SELECT street_id, MAX(density_vpk) FROM road_data GROUP BY street_id"
    ):
        i = id_to_idx.get(street_id)
        if i is not None and max_density is not None:
            peak[i] = max_density

    x = lane_metres(edges)
    # DSF counts an agent as a half while it straddles two streets, so the
    # counts live on a 0.5 grid; snap to it before the ceil, or float error
    # rounds an exact 114 up to 115 and the bound comes out far too tight.
    n = np.ceil(np.round(implied_agents(peak, edges) * 2.0) / 2.0)
    usable = (n >= 2) & (x > 0)
    if not usable.any():
        return float("nan"), 0
    bounds = x[usable] / (n[usable] - 1.0)
    # The bound is strict (L < …), so step just inside it: landing exactly on it
    # would round some edge's capacity down and push it past full.
    estimate = float(bounds.min()) * (1.0 - 1e-9)
    witnesses = int((bounds <= bounds.min() * 1.01).sum())
    return estimate, witnesses


def edge_capacity(edges: list[dict[str, Any]], vehicle_length_m: float) -> np.ndarray:
    """
    Per-edge capacity in vehicles, exactly as DSF computes it:

        capacity = ceil(length_m * nlanes / L)   (>= 1)

    See Road::Road in DynamicalSystemFramework/src/dsf/mobility/Road.cpp. The
    ceil is not cosmetic: on edges shorter than a couple of vehicles it is what
    keeps a single agent from reading as several times "full".
    """
    return np.maximum(1.0, np.ceil(lane_metres(edges) / vehicle_length_m))


def jam_density_vpk(edges: list[dict[str, Any]], vehicle_length_m: float) -> np.ndarray:
    """
    Per-edge density, in the units of ``density_vpk``, at which an edge is full.

    Dividing a stored density by this gives DSF's own ``nAgents / capacity``:
    the implied agent count cancels and what is left is the occupancy the
    simulation itself caps at 1. It sits near ``1000 / vehicle_length_m`` for
    ordinary edges and rises above it for edges short enough that the ceil in
    edge_capacity rounds their capacity up.
    """
    x = lane_metres(edges)
    jam = np.zeros(len(edges), dtype=np.float32)
    ok = x > 0
    jam[ok] = 1000.0 * edge_capacity(edges, vehicle_length_m)[ok] / x[ok]
    return jam


# ── Colour helpers ────────────────────────────────────────────────────────────


def ramp_color(t: float) -> tuple[int, int, int]:
    """
    The colour scale, as (r, g, b), for *t* in [0, 1]: blue → yellow → red.

    Blue rather than green at the low end so the scale stays readable with
    red-green colour vision deficiency: blue↔yellow↔red varies in both hue and
    lightness, which green→yellow→red does not.

    Single source of truth for the scale - the legend and the vectorised
    edge colouring below both go through it (or mirror it exactly).
    """
    t = max(0.0, min(1.0, t))
    if t <= 0.5:
        s = t * 2  # 0 → 1 over the first half:  blue → yellow
        return int(s * 255), int(s * 255), int((1.0 - s) * 255)
    s = (t - 0.5) * 2  # 0 → 1 over the second half: yellow → red
    return 255, int((1.0 - s) * 255), 0


def value_to_qcolor(
    value: float, dmin: float, dmax: float, reversed_: bool = False, alpha: int = 176
) -> QColor:
    """
    Map *value* in [dmin, dmax] to a QColor on the blue→yellow→red scale.
    Pass reversed_=True for red→yellow→blue (used for speed).
    alpha=176 ≈ 0.69 x 255, matching the JS rgba(…, 0.69).

    Kept for single-value use (e.g. tooltips); bulk colouring should use
    edge_colors_for_timestep() instead, which is vectorised.
    """
    rng = dmax - dmin
    t = 0.0 if rng == 0 else (value - dmin) / rng
    t = max(0.0, min(1.0, t))
    if reversed_:
        t = 1.0 - t

    r, g, b = ramp_color(t)
    return QColor(r, g, b, alpha)


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

    # Vectorised mirror of ramp_color(): blue → yellow → red.
    r = np.empty_like(t)
    g = np.empty_like(t)
    b = np.empty_like(t)
    lo = t <= 0.5
    hi = ~lo

    s1 = t[lo] * 2
    r[lo] = s1 * 255
    g[lo] = s1 * 255
    b[lo] = (1.0 - s1) * 255

    s2 = (t[hi] - 0.5) * 2
    r[hi] = 255
    g[hi] = (1.0 - s2) * 255
    b[hi] = 0

    r_list = r.astype(np.uint8).tolist()
    g_list = g.astype(np.uint8).tolist()
    b_list = b.astype(np.uint8).tolist()
    return [QColor(ri, gi, bi, alpha) for ri, gi, bi in zip(r_list, g_list, b_list)]


def observable_row(
    key: str,
    idx: int,
    values: dict[str, np.ndarray],
    jam: np.ndarray | None = None,
) -> np.ndarray:
    """
    Values of observable *key* at timestep *idx*, one entry per edge.

    "density_norm" is derived on the fly from the stored density array divided
    by *jam* (see jam_density_vpk), so no second (T, n_edges) array is kept in
    memory. The result is DSF's own nAgents/capacity, which the simulation caps
    at 1 (an edge cannot hold more agents than its capacity); it is clipped to
    [0, 1] anyway, so a fallback capacity model cannot push a colour off scale.
    """
    src = "density" if key == "density_norm" else key
    arr = values.get(src)
    if arr is None or idx >= len(arr):
        return np.zeros(0, dtype=np.float32)
    row = np.nan_to_num(arr[idx], nan=0.0)
    if key != "density_norm":
        return row
    if jam is None or jam.size != row.size:
        return row
    out = np.divide(row, jam, out=np.zeros_like(row), where=jam > 0)
    return np.clip(out, 0.0, 1.0)


def edge_colors_for_timestep(
    key: str,
    idx: int,
    values: dict[str, np.ndarray],
    domains: dict[str, tuple[float, float]],
    alpha: int = 176,
    jam: np.ndarray | None = None,
) -> list[QColor]:
    """
    Build the QColor list for ONE timestep of ONE observable, on demand.
    Call this from the UI whenever the slider moves / the observable
    changes - it's cheap (O(n_edges)) and replaces precompute_all_colors().

    Pass *jam* (from jam_density_vpk) to enable the "density_norm" observable.
    """
    row = observable_row(key, idx, values, jam)
    if row.size == 0:
        return []
    dmin, dmax = domains.get(key, (0.0, 1.0))
    rev = EDGE_OBSERVABLE_CONFIG.get(key, {}).get("reverseColorScale", False)
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


class LoadCancelled(Exception):
    """Raised out of load_road_data when progress_cb asks it to stop."""


def load_road_data(
    conn: sqlite3.Connection,
    edges: list[dict],
    sim_id: int,
    *,
    vehicle_length: float | None = None,
    progress_cb: Callable[[int], bool] | None = None,
) -> dict[str, Any]:
    """
    Load per-edge time-series data for simulation *sim_id*.

    This is the expensive call in the app - seconds of full-table scan - so it
    takes two optional hooks for callers that run it off the GUI thread:

    progress_cb(rows_read)
        Called every PROGRESS_ROW_INTERVAL rows. Return False to abort, which
        raises LoadCancelled.
    vehicle_length
        Skips estimate_mean_vehicle_length, which scans *every* simulation in
        the file and so gives the same answer for all of them: a caller
        switching simulations within one file can pass the value it already has
        and save that second scan.

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
        "jam_density": np.ndarray,                   # (n_edges,) vpk at capacity
        "vehicle_length_m": float,                   # recovered from the data
        "vehicle_length_source": str,                # how it was arrived at
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

    rows = 0
    for ts, sid, d, s, t, n, q in cur:
        rows += 1
        if (
            progress_cb is not None
            and rows % PROGRESS_ROW_INTERVAL == 0
            and not progress_cb(rows)
        ):
            raise LoadCancelled
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
    # Normalised density is an occupancy fraction, so its scale is fixed rather
    # than data-driven - every edge is measured against its own capacity.
    domains["density_norm"] = (0.0, DENSITY_NORM_SATURATION)

    # Recover the capacity model from the data rather than assuming one - unless
    # the caller already has it for this file.
    if vehicle_length is not None:
        return {
            "datetimes": datetimes,
            "values": values,
            "domains": domains,
            "jam_density": jam_density_vpk(edges, vehicle_length),
            "vehicle_length_m": vehicle_length,
            "vehicle_length_source": "carried over from this database",
        }

    vehicle_length, witnesses = estimate_mean_vehicle_length(conn, edges)
    lo, hi = PLAUSIBLE_VEHICLE_LENGTH_M
    if not (np.isfinite(vehicle_length) and lo <= vehicle_length <= hi):
        source = (
            f"no edge in this run ever filled up — assuming DSF's default "
            f"{DSF_DEFAULT_VEHICLE_LENGTH_M:g} m"
        )
        vehicle_length = DSF_DEFAULT_VEHICLE_LENGTH_M
    else:
        source = f"measured from {witnesses} saturated edge(s)"

    return {
        "datetimes": datetimes,
        "values": values,
        "domains": domains,
        "jam_density": jam_density_vpk(edges, vehicle_length),
        "vehicle_length_m": vehicle_length,
        "vehicle_length_source": source,
    }


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
