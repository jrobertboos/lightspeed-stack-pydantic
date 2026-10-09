from typing import Literal, Optional, Sequence, Set

from pydantic_ai.embeddings import Embedder, EmbeddingSettings

from lightspeed.core.agent.knowledge.sources.base import VectorStore, VectorStoreKnowledgeSource
from lightspeed.core.agent.knowledge.types import KnowledgeMatch


class PgvectorVectorStore(VectorStore):
    """Vector store backed by PostgreSQL with the pgvector extension.

    Not implemented yet — reserved for similarity search over an embedded
    document table (typically via a ``dsn`` from YAML ``config``).
    """

    async def search(
        self, embeddings: Sequence[Sequence[float]], limit: int = 5, threshold: float = 0.0
    ) -> list[KnowledgeMatch]:
        """Search the pgvector table for matches."""
        raise NotImplementedError("PgvectorVectorStore.search is not implemented yet")


class PgvectorKnowledgeSource(VectorStoreKnowledgeSource):
    """Knowledge source that uses pgvector for vector storage.

    Convenience wrapper around :class:`VectorStoreKnowledgeSource` that builds
    a :class:`PgvectorVectorStore` from a DSN. Not implemented yet — register
    a built instance via
    :class:`~lightspeed.core.agent.knowledge.registry.KnowledgeSourceRegistry`
    once the store client exists.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        embedder: Embedder,
        embedding_settings: Optional[EmbeddingSettings] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        mode: Optional[Set[Literal["tool", "auto"]]] = None,
    ) -> None:
        """Build a pgvector-backed knowledge source (store search not implemented yet).

        Args:
            name: Name of this knowledge source.
            embedder: Embedder used to embed search queries.
            embedding_settings: Optional settings to use for the embedding.
            top_k: Maximum number of matches to return.
            score_threshold: Minimum similarity score to return.
            mode: How this source is exposed -- ``'tool'``, ``'auto'``, or both.
                Defaults to ``{'tool'}``.
        """
        super().__init__(
            name=name,
            store=PgvectorVectorStore(),
            embedder=embedder,
            embedding_settings=embedding_settings,
            top_k=top_k,
            score_threshold=score_threshold,
            mode=mode if mode is not None else {"tool"},
        )
