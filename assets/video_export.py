"""
video_export.py
───────────────
Records the map animation to an H.264 MP4, by piping raw frames to ffmpeg.

Why a subprocess
────────────────
System ffmpeg (with libx264) is the only encoder here that produces real H.264:
PySide6's bundled QtMultimedia is LGPL and can encode nothing better than
MPEG-4 Part 2, and imageio / PyAV / OpenCV are not dependencies of this project.
Piping raw frames to `ffmpeg -f rawvideo` costs nothing and adds no dependency.

Why a timer instead of a loop
─────────────────────────────
MapWidget installs freshly downloaded tiles in a *queued* slot, so a blocking
`for` loop over frames would never let a single tile arrive - every frame would
be rendered against whatever basemap happened to be cached when it started.
VideoRecorder is therefore a QTimer-driven pump: it returns to the event loop
between frames, which also keeps the progress dialog alive and Cancel clickable.

Nothing here imports MainWindow; the host injects what the recorder needs as
plain callables, so this module can be exercised without a database or a map.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path

from PySide6.QtCore import QObject, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter

# How long to wait for the visible tiles before recording anyway. A tile whose
# download failed is never retried and never leaves MapWidget._pending, so this
# is a hard wall-clock bound, not a hint.
TILE_WARMUP_TIMEOUT_MS = 8000
TILE_POLL_MS = 100

# ffmpeg is given this long to flush and exit after stdin closes.
ENCODER_CLOSE_TIMEOUT_S = 60


class EncoderError(RuntimeError):
    """Encoding failed; the message carries whatever ffmpeg said about it."""


def ffmpeg_path() -> str | None:
    """Absolute path to the ffmpeg binary, or None if it is not installed."""
    return shutil.which("ffmpeg")


# ── Frame geometry ────────────────────────────────────────────────────────────


def even_size(size: QSize) -> QSize:
    """
    *size* rounded down to even width and height.

    libx264 with yuv420p cannot encode odd dimensions, and a widget grab is
    whatever size the widget happens to be - usually odd.
    """
    return QSize(size.width() - size.width() % 2, size.height() - size.height() % 2)


def fit(image: QImage, size: QSize) -> QImage:
    """
    *image* cropped (or letterboxed onto black) to exactly *size*.

    Every frame of a stream must have the geometry declared when it was opened,
    so the size is pinned from the first frame and everything afterwards is
    forced to match - including frames grabbed after the user resized the window
    mid-recording.
    """
    if image.size() == size:
        return image
    out = QImage(size, QImage.Format_RGB32)
    out.fill(Qt.black)
    p = QPainter(out)
    p.drawImage(0, 0, image)
    p.end()
    return out


# ── Encoder ───────────────────────────────────────────────────────────────────


class FFmpegSink:
    """
    An open ffmpeg process that turns written QImages into an MP4.

    Frames go in as raw BGRA: QWidget.grab().toImage() is Format_RGB32, which in
    memory is B,G,R,A on every little-endian machine, so ffmpeg's `bgra` input
    format consumes the grab verbatim with no per-frame conversion at all.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._proc: subprocess.Popen | None = None
        self._stderr_tail: deque[str] = deque(maxlen=40)
        self._stderr_thread: threading.Thread | None = None

    @property
    def path(self) -> Path:
        return self._path

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def open(self, size: QSize, fps: float) -> None:
        exe = ffmpeg_path()
        if exe is None:
            raise EncoderError("ffmpeg was not found on PATH.")
        if size.width() <= 0 or size.height() <= 0:
            raise EncoderError(
                f"Refusing to record a {size.width()}x{size.height()} frame."
            )

        argv = [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            # Input: the raw widget grabs.
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgra",
            "-video_size",
            f"{size.width()}x{size.height()}",
            "-framerate",
            f"{fps:g}",
            "-i",
            "-",
            # Output. veryfast rather than a slower preset because a write to
            # ffmpeg's stdin blocks the GUI thread whenever x264 falls behind,
            # and at crf 18 on flat map graphics the difference is invisible.
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self._path),
        ]
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise EncoderError(f"Could not start ffmpeg: {exc}") from exc

        # Drain stderr on a thread: ffmpeg deadlocks if it fills the pipe buffer
        # while we are busy writing frames, and this tail is the only diagnostic
        # available when encoding goes wrong.
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._proc.stderr,), daemon=True
        )
        self._stderr_thread.start()

    def write(self, image: QImage) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise EncoderError("Encoder is not open.")
        if image.format() != QImage.Format_RGB32:
            image = image.convertToFormat(QImage.Format_RGB32)
        try:
            # constBits() is a zero-copy view; `image` stays referenced for the
            # duration of the write, which is what keeps that view valid.
            self._proc.stdin.write(image.constBits())
        except (BrokenPipeError, OSError) as exc:
            raise EncoderError(
                self._diagnostic(f"ffmpeg stopped reading: {exc}")
            ) from exc

    def close(self) -> None:
        """Finish the file. Raises EncoderError if ffmpeg did not like it."""
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            returncode = proc.wait(timeout=ENCODER_CLOSE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            self._unlink_partial()
            raise EncoderError(
                "ffmpeg did not finish; the recording was discarded."
            ) from None
        if returncode != 0:
            self._unlink_partial()
            raise EncoderError(
                self._diagnostic(f"ffmpeg exited with status {returncode}.")
            )

    def abort(self) -> None:
        """Tear down without finishing, and remove the partial file. Never raises."""
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except OSError, subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=3)
                except OSError, subprocess.TimeoutExpired:
                    pass
        self._unlink_partial()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _drain_stderr(self, stream) -> None:
        # The stream is handed over rather than read off self._proc, which
        # close() clears from the GUI thread while this one is still running.
        if stream is None:
            return
        try:
            for raw in stream:
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
        except OSError, ValueError:
            pass

    def _diagnostic(self, headline: str) -> str:
        tail = "\n".join(self._stderr_tail)
        return f"{headline}\n\n{tail}" if tail else headline

    def _unlink_partial(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass


# ── Recorder ──────────────────────────────────────────────────────────────────


class RecorderState(Enum):
    IDLE = auto()
    WARMING = auto()
    RECORDING = auto()
    FINISHING = auto()
    DONE = auto()
    FAILED = auto()
    CANCELLED = auto()


class VideoRecorder(QObject):
    """
    Renders a list of timesteps to *sink*, one frame per event-loop turn.

    The host supplies four callables rather than the window itself:
      apply_frame(idx)  - put the UI on timestep idx
      capture()         - QImage of whatever should be recorded
      tiles_pending()   - how many basemap tiles are still in flight
      request_tiles()   - ask for the visible tiles to be fetched
    """

    progress = Signal(int, int)  # frames done, frames total
    finished = Signal(
        bool, str
    )  # ok, path on success / message on failure / "" cancelled

    def __init__(
        self,
        *,
        indices: list[int],
        sink: FFmpegSink,
        fps: float,
        apply_frame: Callable[[int], None],
        capture: Callable[[], QImage],
        tiles_pending: Callable[[], int] | None = None,
        request_tiles: Callable[[], None] | None = None,
        warmup_timeout_ms: int = TILE_WARMUP_TIMEOUT_MS,
        parent: QObject | None = None,
    ):
        super().__init__(parent)
        self._indices = list(indices)
        self._sink = sink
        self._fps = fps
        self._apply_frame = apply_frame
        self._capture = capture
        self._tiles_pending = tiles_pending
        self._request_tiles = request_tiles
        self._warmup_timeout_ms = warmup_timeout_ms

        self._state = RecorderState.IDLE
        self._done = 0
        self._size: QSize | None = None
        self._warm_deadline = 0.0
        self._cancel_requested = False
        self._in_tick = False

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._tick)

    # ── Public API ───────────────────────────────────────────────────────────

    def is_active(self) -> bool:
        return self._state in (
            RecorderState.WARMING,
            RecorderState.RECORDING,
            RecorderState.FINISHING,
        )

    def start(self) -> None:
        if self._state is not RecorderState.IDLE:
            return
        if not self._indices:
            self._teardown(
                RecorderState.FAILED, "Nothing to record: the time window is empty."
            )
            return
        self._state = RecorderState.WARMING
        self._warm_deadline = time.monotonic() + self._warmup_timeout_ms / 1000.0
        if self._request_tiles is not None:
            self._request_tiles()
        self._timer.start(TILE_POLL_MS)

    def cancel(self) -> None:
        """Ask to stop. Idempotent, and safe to call from a signal handler."""
        if self.is_active():
            self._cancel_requested = True

    def finish_now(self) -> None:
        """
        Tear down synchronously, for when there will be no next tick - closing
        the window kills the event loop, and without this the ffmpeg process
        would outlive the app holding a half-written file.
        """
        if self.is_active():
            self._teardown(RecorderState.CANCELLED)

    # ── Frame pump ───────────────────────────────────────────────────────────

    def _tick(self) -> None:
        # A modal QProgressDialog pumps the event loop from setValue(), which can
        # re-enter this slot from inside progress.emit() and interleave two frame
        # writes into the stream.
        if self._in_tick:
            return
        self._in_tick = True
        try:
            if self._cancel_requested:
                self._teardown(RecorderState.CANCELLED)
                return
            try:
                if self._state is RecorderState.WARMING:
                    self._tick_warming()
                elif self._state is RecorderState.RECORDING:
                    self._tick_recording()
                elif self._state is RecorderState.FINISHING:
                    self._sink.close()
                    self._teardown(RecorderState.DONE)
            except EncoderError as exc:
                self._teardown(RecorderState.FAILED, str(exc))
            except Exception as exc:  # noqa: BLE001 - never leave ffmpeg running
                self._teardown(RecorderState.FAILED, f"Recording failed: {exc}")
        finally:
            self._in_tick = False

    def _tick_warming(self) -> None:
        pending = self._tiles_pending() if self._tiles_pending is not None else 0
        if pending and time.monotonic() < self._warm_deadline:
            self._timer.start(TILE_POLL_MS)
            return

        # First frame also fixes the geometry for the whole stream.
        self._apply_frame(self._indices[0])
        first = self._capture()
        self._size = even_size(first.size())
        self._sink.open(self._size, self._fps)
        self._sink.write(fit(first, self._size))
        self._done = 1
        self._state = RecorderState.RECORDING
        self.progress.emit(self._done, len(self._indices))
        self._timer.start(0)

    def _tick_recording(self) -> None:
        if self._done >= len(self._indices):
            self._state = RecorderState.FINISHING
            self._timer.start(0)
            return
        assert self._size is not None
        self._apply_frame(self._indices[self._done])
        self._sink.write(fit(self._capture(), self._size))
        self._done += 1
        self.progress.emit(self._done, len(self._indices))
        # Zero-interval, but still a return to the event loop: queued tile slots
        # run, the progress dialog repaints, Cancel stays clickable.
        self._timer.start(0)

    def _teardown(self, state: RecorderState, message: str = "") -> None:
        self._timer.stop()
        if state is RecorderState.DONE:
            self.finished.emit(True, str(self._sink.path))
        else:
            self._sink.abort()
            self.finished.emit(False, message)
        self._state = state
