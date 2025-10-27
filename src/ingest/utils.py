from __future__ import annotations

import re


_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9\s-]")
_SEPARATOR_PATTERN = re.compile(r"[\s-]+")


def slugify(text: str, fallback: str = "section") -> str:
    """Create a filesystem-friendly slug from arbitrary text."""

    slug = _SLUG_PATTERN.sub("", text).strip().lower()
    slug = _SEPARATOR_PATTERN.sub("-", slug)
    return slug or fallback


__all__ = ["slugify"]
