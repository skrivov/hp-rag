from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Optional


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class EventFactory:
    """Utility to build structured progress events."""

    run_id: str

    def event(
        self,
        *,
        phase: str,
        type: str,
        name: str,
        summary: str,
        detail: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        step_id: str | None = None,
        parent_step_id: str | None = None,
    ) -> Dict[str, Any]:
        return {
            "version": 1,
            "ts": _ts(),
            "run_id": self.run_id,
            "phase": phase,
            "type": type,
            "name": name,
            "summary": summary,
            "detail": detail or {},
            "metrics": metrics or {},
            "severity": severity,
            "step_id": step_id,
            "parent_step_id": parent_step_id,
        }

    def step(
        self,
        *,
        phase: str,
        name: str,
        summary: str,
        detail: Optional[Dict[str, Any]] = None,
        parent_step_id: str | None = None,
    ) -> "StepTracker":
        return StepTracker(
            factory=self,
            phase=phase,
            name=name,
            summary=summary,
            detail=detail or {},
            parent_step_id=parent_step_id,
        )


@dataclass(slots=True)
class StepTracker:
    factory: EventFactory
    phase: str
    name: str
    summary: str
    detail: Dict[str, Any] = field(default_factory=dict)
    parent_step_id: str | None = None
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=perf_counter)

    def start_event(self) -> Dict[str, Any]:
        return self.factory.event(
            phase=self.phase,
            type="start",
            name=self.name,
            summary=self.summary,
            detail=self.detail,
            step_id=self.step_id,
            parent_step_id=self.parent_step_id,
        )

    def end_event(self, detail: Optional[Dict[str, Any]] = None, summary: str | None = None) -> Dict[str, Any]:
        duration_ms = int((perf_counter() - self.start_time) * 1000)
        return self.factory.event(
            phase=self.phase,
            type="end",
            name=self.name,
            summary=summary or f"{self.summary} — done",
            detail=detail or {},
            metrics={"latency_ms": duration_ms},
            step_id=self.step_id,
            parent_step_id=self.parent_step_id,
        )

    def progress_event(self, summary: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.factory.event(
            phase=self.phase,
            type="progress",
            name=self.name,
            summary=summary,
            detail=detail or {},
            step_id=self.step_id,
            parent_step_id=self.parent_step_id,
        )

    def error_event(self, message: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.factory.event(
            phase=self.phase,
            type="error",
            name=self.name,
            summary=message,
            detail=detail or {},
            severity="error",
            step_id=self.step_id,
            parent_step_id=self.parent_step_id,
        )


__all__ = ["EventFactory", "StepTracker"]
