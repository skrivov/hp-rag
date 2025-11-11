from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from .settings import Settings

if TYPE_CHECKING:
    from .sse import EventBus

SystemName = Literal["hp-rag", "rag"]


class RunCreate(BaseModel):
    """Client payload for launching a chat run."""

    system: SystemName = Field(description="Retrieval strategy to use: hp-rag or rag")
    message: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    run_id: str | None = Field(default=None, description="Optional client-provided run_id for idempotency")
    tenant_id: str | None = Field(default=None, description="Optional tenant context for this run")

    @field_validator("system")
    @classmethod
    def _normalize_system(cls, value: str) -> str:
        normalized = value.lower()
        if normalized not in {"hp-rag", "rag"}:
            raise ValueError("system must be 'hp-rag' or 'rag'")
        return normalized


class RunConfig(BaseModel):
    """Fully-hydrated configuration for executing a run."""

    run_id: str
    system: SystemName
    message: str
    top_k: int
    metadata: Dict[str, Any]
    system_prompt: str
    prompt_template: str | None
    temperature: float
    tenant_id: str | None = None

    @classmethod
    def from_request(cls, payload: RunCreate, settings: Settings) -> "RunConfig":
        run_id = payload.run_id or str(uuid.uuid4())
        top_k = payload.top_k or settings.default_top_k
        return cls(
            run_id=run_id,
            system=payload.system,
            message=payload.message.strip(),
            top_k=top_k,
            metadata=payload.metadata or {},
            system_prompt=settings.system_prompt,
            prompt_template=settings.prompt_template,
            temperature=settings.temperature,
            tenant_id=payload.tenant_id,
        )


class TokenEvent(BaseModel):
    text: str


class ProgressEvent(BaseModel):
    phase: str
    type: str
    name: str
    summary: str
    detail: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class DoneEvent(BaseModel):
    finish_reason: str
    usage: Dict[str, Any]
    timings: Dict[str, Any]
    error: Dict[str, Any] | None = None


@dataclass(slots=True)
class RunState:
    """In-memory tracking for active SSE runs."""

    config: RunConfig
    bus: "EventBus"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tokens: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    done_payload: dict[str, Any] | None = None
    error: str | None = None
    finished: bool = False

    def append_token(self, text: str) -> None:
        self.tokens.append(text)

    def append_event(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.config.run_id,
            "system": self.config.system,
            "tenant_id": self.config.tenant_id,
            "created_at": self.created_at.isoformat(),
            "tokens": "".join(self.tokens),
            "events": self.events,
            "done": self.done_payload,
            "error": self.error,
        }


__all__ = [
    "DoneEvent",
    "ProgressEvent",
    "RunConfig",
    "RunCreate",
    "RunState",
    "SystemName",
    "TokenEvent",
]
