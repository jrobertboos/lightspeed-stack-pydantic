from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Set

from lightspeed.core.agent.knowledge.types import KnowledgeMatch
from pydantic_ai.embeddings import Embedder, EmbeddingSettings

@dataclass
class KnowledgeSource(ABC):
    """Knowledge source to search, with its own exposure mode.

    `mode` lives here (rather than on whatever capability attaches this
    source) so each source controls how *it* is exposed, independent of how
    many other sources a `Knowledge` capability combines it with.
    """

    name: str
    """Name of the knowledge source."""

    mode: Set[Literal["tool", "auto"]] = field(default_factory=lambda: {"tool"}, kw_only=True)
    """
    `'tool'`: exposes a model-callable `search_<name>` tool for this source.
    `'auto'`: searches the latest user prompt automatically and injects this
    source's matches as bounded, delimited context on every model request.
    """

    @abstractmethod
    async def search(self, prompt: str) -> list[KnowledgeMatch]:
        """Search the knowledge source for matches."""


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
