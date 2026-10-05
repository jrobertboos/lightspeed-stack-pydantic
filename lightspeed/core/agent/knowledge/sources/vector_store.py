from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from pydantic_ai.embeddings import Embedder, EmbeddingSettings

from lightspeed.core.agent.knowledge.sources.base import KnowledgeMatch, KnowledgeSource


class VectorStore(ABC):
    """Vector store to search by embedding vector."""

    @abstractmethod
    async def search(self, embeddings: Sequence[Sequence[float]], limit: int = 5, threshold: float = 0.0) -> list[KnowledgeMatch]:
        """Search the vector store for matches.

        Args:
            embeddings: Query embedding vectors.
            limit: The maximum number of matches to return.
            threshold: The minimum similarity score to return.
        """
        ...


@dataclass
class VectorStoreKnowledgeSource(KnowledgeSource):
    """Knowledge source that embeds queries and searches a :class:`VectorStore`."""

    store: VectorStore
    """Vector store to search."""

    embedder: Embedder
    """Embedder to use for the search."""

    embedding_settings: Optional[EmbeddingSettings] = None
    """Settings to use for the embedding."""

    top_k: int = 5
    """Maximum number of matches to return."""

    score_threshold: float = 0.0
    """Minimum similarity score to return."""

    async def search(self, query: str) -> list[KnowledgeMatch]:
        """Search the vector store for matches."""
        embedding_result = await self.embedder.embed_query(query, settings=self.embedding_settings)
        return await self.store.search(embedding_result.embeddings, limit=self.top_k, threshold=self.score_threshold)
