"""Shared data contracts passed between pipeline modules.

Every module depends ONLY on this file (never on another module), which keeps
the modules independently testable and swappable.

Coordinate convention: bounding boxes are (x1, y1, x2, y2) in ORIGINAL frame
pixel coordinates, origin at top-left.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class StreamError(RuntimeError):
    """Raised when a video source cannot be opened or recovered."""


class ModelLoadError(RuntimeError):
    """Raised when the model file/backend cannot be initialised."""


@dataclass(frozen=True)
class Frame:
    """One raw decoded video frame."""
    frame_id: int
    camera_id: str
    timestamp_ns: int
    image: np.ndarray  # H x W x 3, BGR, uint8

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass
class PreprocessedFrame:
    """Model-ready tensor plus the geometry needed to map outputs back."""
    source: Frame
    tensor: np.ndarray            # 3 x H x W, float32, normalised (CHW)
    scale: float                  # resize ratio applied to the original frame
    pad: Tuple[int, int]          # (pad_left, pad_top) added by letterboxing


@dataclass
class Detection:
    bbox: BBox
    class_id: int
    label: str
    confidence: float
    track_id: Optional[int] = None


@dataclass
class InferenceResult:
    frame_id: int
    camera_id: str
    timestamp_ns: int
    detections: List[Detection] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class Alert:
    alert_id: str
    rule_id: str
    severity: Severity
    camera_id: str
    timestamp_ns: int
    message: str
    detections: List[Detection] = field(default_factory=list)
    snapshot_path: Optional[str] = None


@dataclass
class StreamHealth:
    connected: bool
    fps: float
    dropped_frames: int
    last_frame_age_ms: float
    reconnect_count: int
