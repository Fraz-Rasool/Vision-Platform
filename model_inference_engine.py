"""ModelInferenceEngine: runs the detector and post-processes raw outputs.

Owns: model lifecycle, batching, confidence filtering, NMS, label mapping,
coordinate restoration. Backend (ONNX Runtime / TensorRT / TFLite) is hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .contracts import Detection, InferenceResult, PreprocessedFrame


@dataclass(frozen=True)
class InferenceConfig:
    model_path: str
    backend: str = "onnxruntime"             # "onnxruntime" | "tensorrt" | "tflite"
    device: str = "cpu"                       # "cpu" | "cuda:0"
    conf_threshold: float = 0.40
    iou_threshold: float = 0.50
    max_detections: int = 100
    max_batch_size: int = 4
    class_names: Sequence[str] = field(default_factory=tuple)


class ModelInferenceEngine:
    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._loaded = False

    def load(self) -> None:
        """Load weights/engine. Raises ModelLoadError on failure."""
        raise NotImplementedError

    def warmup(self, iterations: int = 3) -> None:
        """Dummy passes so the first real frame doesn't pay init cost."""
        raise NotImplementedError

    def infer(self, batch: Sequence[PreprocessedFrame]) -> List[InferenceResult]:
        """Run the model; return one InferenceResult per input, same order."""
        raise NotImplementedError

    def _postprocess(self, raw_output: np.ndarray,
                     pre: PreprocessedFrame) -> List[Detection]:
        """Decode raw tensor -> confidence filter -> NMS -> restore bbox -> Detection."""
        raise NotImplementedError

    @staticmethod
    def non_max_suppression(boxes: np.ndarray, scores: np.ndarray,
                            iou_threshold: float) -> List[int]:
        """Greedy NMS. boxes: Nx4 (x1,y1,x2,y2); returns kept indices, best first."""
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-9)
            order = rest[iou <= iou_threshold]
        return keep

    def metrics(self) -> Dict[str, float]:
        """Rolling stats: mean/p95 latency_ms, throughput_fps."""
        raise NotImplementedError

    def unload(self) -> None:
        """Release model memory / GPU context."""
        raise NotImplementedError
