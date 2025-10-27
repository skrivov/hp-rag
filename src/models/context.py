from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(slots=True)
class RetrievedContext:
    """Container for retrieved context snippets prior to LLM querying."""

    path: str
    title: str
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None


__all__ = ["RetrievedContext"]
