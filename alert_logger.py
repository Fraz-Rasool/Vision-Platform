"""AlertLogger: turns detections into alerts and durable log records.

Owns: rule evaluation, cooldown/de-duplication, dispatch to notification sinks,
local persistence (offline-first), and later sync to the central server.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .contracts import Alert, InferenceResult, Severity


@dataclass(frozen=True)
class AlertRule:
    rule_id: str
    target_labels: Sequence[str]
    min_confidence: float = 0.60
    min_count: int = 1
    cooldown_s: float = 30.0
    severity: Severity = Severity.WARNING


class AlertSink(ABC):
    """Notification channel (email, push, webhook, MQTT, ...)."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Deliver the alert. Return True on success."""


class AlertLogger:
    def __init__(self, rules: Sequence[AlertRule], sinks: Sequence[AlertSink],
                 local_db_path: str = "events.db",
                 central_db_uri: Optional[str] = None) -> None:
        self.rules = list(rules)
        self.sinks = list(sinks)
        self.local_db_path = local_db_path
        self.central_db_uri = central_db_uri
        self._last_fired_ns: Dict[str, int] = {}

    def evaluate(self, result: InferenceResult) -> List[Alert]:
        """Apply every rule to one result; honour per-camera cooldown."""
        alerts: List[Alert] = []
        for rule in self.rules:
            hits = [d for d in result.detections
                    if d.label in rule.target_labels and d.confidence >= rule.min_confidence]
            if len(hits) < rule.min_count:
                continue
            key = f"{rule.rule_id}:{result.camera_id}"
            last = self._last_fired_ns.get(key)
            if last is not None and (result.timestamp_ns - last) < rule.cooldown_s * 1e9:
                continue
            self._last_fired_ns[key] = result.timestamp_ns
            alerts.append(Alert(
                alert_id=str(uuid.uuid4()), rule_id=rule.rule_id,
                severity=rule.severity, camera_id=result.camera_id,
                timestamp_ns=result.timestamp_ns,
                message=f"{len(hits)} x {'/'.join(rule.target_labels)} detected",
                detections=hits))
        return alerts

    def dispatch(self, alert: Alert) -> bool:
        """Send to all sinks; True if at least one succeeded. Failures are queued."""
        raise NotImplementedError

    def log_detections(self, result: InferenceResult) -> None:
        """Append detections to the local store (batched, non-blocking)."""
        raise NotImplementedError

    def sync_pending(self, batch_size: int = 100) -> int:
        """Push unsynced rows to central DB with retry; return rows synced."""
        raise NotImplementedError

    def flush(self) -> None:
        """Force pending writes to disk (call on shutdown)."""
        raise NotImplementedError
