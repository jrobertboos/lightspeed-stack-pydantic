"""Pluggable vector-store protocol for the ``Knowledge`` capability.

:class:`VectorStore` is the seam a concrete backend (pgvector, FAISS, ...)
implements; none exists in this rewrite yet -- see
:class:`~lightspeed.core.agent.knowledge.registry.KnowledgeStoreRegistry` for
how a configured source gets one. :class:`InMemoryVectorStore` below is a
real, working reference implementation for tests and local development, in
the same spirit as :class:`pydantic_ai_harness.memory.InMemoryStore`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class KnowledgeMatch:
    """One scored knowledge-search result."""

    content: str
    """The matched chunk's text."""

    score: float
    """Similarity score for this match; higher is more relevant."""

    source: str | None = None
    """Optional citation identifier (e.g. a document URI), for grounding."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Optional backend-specific metadata carried alongside the match."""


@runtime_checkable
class VectorStore(Protocol):
    """Async similarity search over a backing vector database.

    Implement this for a concrete backend (pgvector, FAISS, ...). Nothing in
    this rewrite builds one from
    :class:`~lightspeed.app.models.config.KnowledgeSourceConfiguration` yet --
    register an instance with
    :class:`~lightspeed.core.agent.knowledge.registry.KnowledgeStoreRegistry`
    instead.
    """

    async def search(self, embedding: Sequence[float], *, limit: int) -> list[KnowledgeMatch]:
        """Return the `limit` closest matches to `embedding`, best match first."""
        ...  # pragma: no cover


class InMemoryVectorStore:
    """Process-lifetime :class:`VectorStore` computing cosine similarity in Python.

    A real reference implementation -- not a stub -- meant for tests and
    local development. There is no indexing, persistence, or performance
    guarantee suitable for production use.
    """

    def __init__(self) -> None:
        self._records: list[tuple[list[float], KnowledgeMatch]] = []

    def add(
        self,
        content: str,
        embedding: Sequence[float],
        *,
        source: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Add one document embedding to the store.

        Parameters:
            content: The chunk's text, returned verbatim in matches.
            embedding: The chunk's embedding vector.
            source: Optional citation identifier stored alongside the match.
            metadata: Optional metadata stored alongside the match.
        """
        match = KnowledgeMatch(content=content, score=0.0, source=source, metadata=dict(metadata or {}))
        self._records.append((list(embedding), match))

    async def search(self, embedding: Sequence[float], *, limit: int) -> list[KnowledgeMatch]:
        """Return the `limit` stored records most cosine-similar to `embedding`.

        Raises:
            ValueError: If `limit` is not positive, or a stored embedding's
                dimension doesn't match `embedding`'s.
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        scored = [
            (_cosine_similarity(embedding, stored_embedding), match) for stored_embedding, match in self._records
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            KnowledgeMatch(content=match.content, score=score, source=match.source, metadata=match.metadata)
            for score, match in scored[:limit]
        ]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"embedding dimension mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
