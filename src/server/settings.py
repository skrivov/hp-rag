from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv(override=False)


DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context."
)


class Settings(BaseModel):
    """Runtime configuration for the FastAPI server."""

    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "dev"))
    answer_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    selector_model: str = Field(
        default_factory=lambda: os.getenv("SELECTOR_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    embedding_model: str | None = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.0")))
    default_top_k: int = Field(default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "5")))
    sqlite_db_path: Path = Field(default_factory=lambda: Path(os.getenv("SQLITE_DB", "artifacts/hyperlink.db")))
    faiss_index_path: Path = Field(
        default_factory=lambda: Path(os.getenv("FAISS_INDEX_DIR", "artifacts/faiss_index"))
    )
    uploads_dir: Path = Field(default_factory=lambda: Path(os.getenv("UPLOADS_DIR", "artifacts/uploads")))
    document_queue_size: int = Field(default_factory=lambda: int(os.getenv("DOCUMENT_QUEUE_SIZE", "4")))
    runs_db_path: Path = Field(default_factory=lambda: Path(os.getenv("RUNS_DB_PATH", "artifacts/server_runs.db")))
    run_ttl_seconds: int | None = Field(
        default_factory=lambda: int(os.getenv("RUN_TTL_SECONDS", "86400"))
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:5174"]
    )
    sse_heartbeat_interval: float = Field(default_factory=lambda: float(os.getenv("SSE_HEARTBEAT_SECONDS", "15")))
    allow_dev_params: bool = Field(
        default_factory=lambda: os.getenv("ALLOW_DEV_STREAM_PARAMS", "true").lower() not in {"0", "false"}
    )
    system_prompt: str = Field(default_factory=lambda: os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    prompt_template: str | None = Field(default=None)
    hp_max_sections: int = Field(default_factory=lambda: int(os.getenv("HP_MAX_SECTIONS", "8")))
    hp_toc_limit: int = Field(default_factory=lambda: int(os.getenv("HP_TOC_LIMIT", "200")))
    stream_timeout_seconds: float = Field(default_factory=lambda: float(os.getenv("STREAM_TIMEOUT_SECONDS", "120")))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    model_config = {
        "frozen": True,
    }

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> List[str]:
        if value is None:
            return ["http://localhost:5173", "http://localhost:5174"]
        if isinstance(value, str):
            if not value.strip():
                return ["http://localhost:5173", "http://localhost:5174"]
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return list(value)

    @field_validator("openai_api_key")
    @classmethod
    def _require_api_key(cls, value: str) -> str:
        if not value:
            raise ValueError("OPENAI_API_KEY must be set in the environment for streaming APIs.")
        return value

    @field_validator("run_ttl_seconds", mode="before")
    @classmethod
    def _normalize_ttl(cls, value: object) -> int | None:
        try:
            ttl = int(value)
        except (TypeError, ValueError):
            return None
        return ttl if ttl > 0 else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance for FastAPI dependency injection."""

    return Settings()


__all__ = ["Settings", "get_settings", "DEFAULT_SYSTEM_PROMPT"]
