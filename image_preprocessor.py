"""ImagePreprocessor: converts raw frames into model-ready tensors.

Pure and stateless apart from config => trivially unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .contracts import BBox, Frame, PreprocessedFrame


@dataclass(frozen=True)
class PreprocessConfig:
    target_size: Tuple[int, int] = (640, 640)          # (width, height)
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    letterbox: bool = True                              # keep aspect ratio
    pad_value: int = 114
    bgr_to_rgb: bool = True


class ImagePreprocessor:
    def __init__(self, config: PreprocessConfig = PreprocessConfig()) -> None:
        self.config = config

    def process(self, frame: Frame) -> PreprocessedFrame:
        """Resize/pad -> colour convert -> scale to [0,1] -> normalise -> CHW float32."""
        raise NotImplementedError

    def process_batch(self, frames: Sequence[Frame]) -> List[PreprocessedFrame]:
        """Preprocess several frames (same config); preserves input order."""
        return [self.process(f) for f in frames]

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Return (resized_padded_image, scale, (pad_left, pad_top))."""
        raise NotImplementedError

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """uint8 HWC -> float32 CHW using config mean/std."""
        raise NotImplementedError

    @staticmethod
    def restore_bbox(bbox: BBox, pre: PreprocessedFrame) -> BBox:
        """Map a model-space box back to original-frame pixel coordinates."""
        pad_x, pad_y = pre.pad
        x1, y1, x2, y2 = bbox
        x1, x2 = (x1 - pad_x) / pre.scale, (x2 - pad_x) / pre.scale
        y1, y2 = (y1 - pad_y) / pre.scale, (y2 - pad_y) / pre.scale
        w, h = pre.source.width, pre.source.height
        return (max(0.0, min(x1, w)), max(0.0, min(y1, h)),
                max(0.0, min(x2, w)), max(0.0, min(y2, h)))
