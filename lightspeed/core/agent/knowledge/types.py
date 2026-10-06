from dataclasses import dataclass, field
from typing import Any, Mapping
import uuid


@dataclass(frozen=True)
class KnowledgeMatch:
    """One scored knowledge-search result."""

    content: str
    """The matched chunk's text."""

    score: float
    """Similarity score for this match; higher is more relevant."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this match."""

    source: str | None = None
    """Optional citation identifier (e.g. a document URI), for grounding."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Optional backend-specific metadata carried alongside the match."""
