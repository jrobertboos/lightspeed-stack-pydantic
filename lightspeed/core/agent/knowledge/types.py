from dataclasses import dataclass, field
from typing import Any, Literal, Mapping
import uuid

KnowledgeReadinessStatus = Literal["ready", "degraded", "not_ready"]
"""Common readiness codes; backends may use additional `status` strings."""


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


@dataclass(frozen=True)
class KnowledgeReadiness:
    """Operational readiness of a knowledge source.

    ``status`` is an open-ended code: use ``"ready"``, ``"degraded"``, or
    ``"not_ready"`` for the usual cases, or a backend-specific value (for
    example ``"index_empty"``). Structured diagnostics belong in ``details``.
    """

    status: KnowledgeReadinessStatus
    """Readiness code; commonly one of ``KnowledgeReadinessStatus``."""

    message: str = ""
    """Human-readable summary for operators and logs."""