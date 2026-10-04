"""VisionPipeline: orchestrates the four modules. Contains no CV logic itself."""
from __future__ import annotations

from .alert_logger import AlertLogger
from .data_ingestion import DataIngestion
from .image_preprocessor import ImagePreprocessor
from .model_inference_engine import ModelInferenceEngine


class VisionPipeline:
    def __init__(self, ingestion: DataIngestion, preprocessor: ImagePreprocessor,
                 engine: ModelInferenceEngine, alerts: AlertLogger) -> None:
        self.ingestion = ingestion
        self.preprocessor = preprocessor
        self.engine = engine
        self.alerts = alerts

    def run(self) -> None:
        self.engine.load()
        self.engine.warmup()
        try:
            for frame in self.ingestion.frames():
                pre = self.preprocessor.process(frame)
                result = self.engine.infer([pre])[0]
                self.alerts.log_detections(result)
                for alert in self.alerts.evaluate(result):
                    self.alerts.dispatch(alert)
        finally:
            self.alerts.flush()
            self.engine.unload()
