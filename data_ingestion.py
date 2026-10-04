"""DataIngestion: acquires frames from a camera/stream and hides transport details.

Responsibilities : connect, decode, timestamp, drop-oldest buffering, reconnect.
NOT responsible  : resizing, normalisation, inference, alerting.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .contracts import Frame, StreamError, StreamHealth


class DataIngestion(ABC):
    def __init__(self, source_uri: str, camera_id: str,
                 target_fps: int = 15, buffer_size: int = 4) -> None:
        self.source_uri = source_uri
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.buffer_size = buffer_size      # small buffer => bounded latency/memory
        self._reconnects = 0

    @abstractmethod
    def connect(self) -> bool:
        """Open the source. Returns True on success, False otherwise."""

    @abstractmethod
    def read_frame(self, timeout_s: float = 1.0) -> Optional[Frame]:
        """Return the newest frame, or None on timeout. Must not block forever."""

    @abstractmethod
    def health(self) -> StreamHealth:
        """Snapshot of stream health for monitoring."""

    @abstractmethod
    def release(self) -> None:
        """Free sockets / decoder resources. Safe to call twice."""

    def reconnect(self, max_attempts: int = 5, backoff_s: float = 1.0) -> bool:
        """Exponential-backoff reconnect. Returns True once connected."""
        self.release()
        for attempt in range(max_attempts):
            time.sleep(backoff_s * (2 ** attempt))
            if self.connect():
                self._reconnects += 1
                return True
        return False

    def frames(self) -> Iterator[Frame]:
        """Endless frame generator; raises StreamError if recovery fails."""
        if not self.connect():
            raise StreamError(f"cannot open {self.source_uri}")
        try:
            while True:
                frame = self.read_frame()
                if frame is None and not self.reconnect():
                    raise StreamError(f"stream lost: {self.source_uri}")
                if frame is not None:
                    yield frame
        finally:
            self.release()


class RTSPIngestion(DataIngestion):
    """RTSP/H.264 source. Backend (OpenCV+FFmpeg or GStreamer) is an implementation detail."""

    def connect(self) -> bool:
        raise NotImplementedError

    def read_frame(self, timeout_s: float = 1.0) -> Optional[Frame]:
        raise NotImplementedError

    def health(self) -> StreamHealth:
        raise NotImplementedError

    def release(self) -> None:
        raise NotImplementedError
