"""Embed :class:`KnowledgeChunk` text with a pydantic-ai :class:`Embedder`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from pydantic_ai.embeddings import Embedder, EmbeddingSettings

from lightspeed.core.agent.knowledge.types import KnowledgeChunk


async def embed_chunk(
    chunk: KnowledgeChunk,
    embedder: Embedder,
    *,
    settings: Optional[EmbeddingSettings] = None,
) -> Sequence[float]:
    """Embed one chunk as a document (not a query).

    Args:
        chunk: Chunk whose ``content`` should be embedded.
        embedder: pydantic-ai embedder, e.g. ``Embedder("openai:text-embedding-3-small")``.
        settings: Optional per-call embedding settings.

    Returns:
        The embedding vector for `chunk`.
    """
    result = await embedder.embed_documents(chunk.content, settings=settings)
    return result.embeddings[0]


async def embed_chunks(
    chunks: Sequence[KnowledgeChunk],
    embedder: Embedder,
    *,
    settings: Optional[EmbeddingSettings] = None,
) -> list[Sequence[float]]:
    """Embed many chunks in one :meth:`Embedder.embed_documents` call.

    Args:
        chunks: Chunks whose ``content`` should be embedded, in order.
        embedder: pydantic-ai embedder used for document embeddings.
        settings: Optional per-call embedding settings.

    Returns:
        One embedding vector per chunk, in the same order. Empty when
        `chunks` is empty (the model is not called).
    """
    if not chunks:
        return []
    result = await embedder.embed_documents(
        [chunk.content for chunk in chunks],
        settings=settings,
    )
    return list(result.embeddings)
