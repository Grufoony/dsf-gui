"""
sim_loader.py
─────────────
Reads one simulation's time series on a worker thread.

Why off the GUI thread
──────────────────────
load_road_data is a full scan of an unindexed road_data table - about six
seconds on a 7-million-row file. Run on the GUI thread it freezes everything,
and Qt never gets an event-loop turn to finish hiding the widget that started
the load, so the simulation picker stays painted on screen as a ghost until the
read completes. Moving the read here fixes both at once.

Keeping this in its own module keeps Qt threading out of database.py, which
otherwise touches Qt only for QColor.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from .database import LoadCancelled, load_global_data, load_road_data


class SimulationLoader(QObject):
    """
    Loads one simulation, reporting progress and honouring cancel.

    Emits exactly one `done`, whatever happens. On cancel or failure the result
    is None and nothing in the caller's state should change - the caller applies
    the result only on success, so a cancelled switch leaves the simulation that
    was already on screen untouched.
    """

    progress = Signal(int)  # rows read so far
    done = Signal(object, str)  # (result | None, error message; "" if cancelled)

    def __init__(
        self,
        db_path: Path | str,
        sim_id: int,
        edges: list[dict],
        vehicle_length: float | None = None,
    ):
        super().__init__()
        self._db_path = str(db_path)
        self._sim_id = sim_id
        self._edges = edges
        self._vehicle_length = vehicle_length
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Ask the read to stop. Safe to call from the GUI thread."""
        self._cancel.set()

    def run(self) -> None:
        conn: sqlite3.Connection | None = None
        try:
            # Opened here, not handed in: sqlite3 connections belong to the
            # thread that created them.
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            bundle = load_road_data(
                conn,
                self._edges,
                self._sim_id,
                vehicle_length=self._vehicle_length,
                progress_cb=self._on_rows,
            )
            global_data = load_global_data(conn, self._sim_id)
        except LoadCancelled:
            self.done.emit(None, "")
            return
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            self.done.emit(None, str(exc))
            return
        finally:
            if conn is not None:
                conn.close()

        result: dict[str, Any] = {
            "sim_id": self._sim_id,
            "bundle": bundle,
            "global_data": global_data,
        }
        self.done.emit(result, "")

    def _on_rows(self, rows: int) -> bool:
        self.progress.emit(rows)
        return not self._cancel.is_set()
