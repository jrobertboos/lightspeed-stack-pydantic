from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pydantic_ai.embeddings import Embedder, EmbeddingResult, EmbeddingSettings

from lightspeed.core.agent.knowledge.capability import KnowledgeMatch, KnowledgeSource


class VectorStore(ABC):
    """Vector store to search by embedding vector."""

    @abstractmethod
    async def search(self, embedding: EmbeddingResult, limit: int = 5) -> list[KnowledgeMatch]:
        """Search the vector store for matches.

        Args:
            embedding: Query embedding vector.
            limit: The maximum number of matches to return.
        """
        ...


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

    async def search(self, query: str) -> list[KnowledgeMatch]:
        """Search the vector store for matches."""
        embedding_result = await self.embedder.embed_query(query, settings=self.embedding_settings)
        return await self.store.search(embedding_result, limit=self.top_k)
