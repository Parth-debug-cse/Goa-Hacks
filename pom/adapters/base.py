"""Base protocol and data models for search adapters (§3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

MatchConfidenceHint = Literal["exact", "visual"]


@dataclass
class Candidate:
    url: str
    title: str | None = None
    source_engine: str = "unknown"
    thumbnail: str | None = None
    match_confidence_hint: MatchConfidenceHint = "visual"
    provenance_id: str = ""
    search_hop: int = 1
    discovered_via: dict[str, Any] | None = None


@dataclass
class SearchResponse:
    engine: str
    candidates: list[Candidate] = field(default_factory=list)
    provenance_id: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)
    warning: str | None = None


@runtime_checkable
class SearchAdapter(Protocol):
    """Uniform protocol for all reverse-image and identity search adapters (AH-6)."""
    
    def search(self, query_or_image: Any, **kwargs: Any) -> SearchResponse:
        ...
