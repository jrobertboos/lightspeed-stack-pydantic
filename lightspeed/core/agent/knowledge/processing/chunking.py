"""Split a :class:`KnowledgeDocument` into embeddable chunks with Docling."""

from __future__ import annotations

from typing import Any, Optional

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter

from lightspeed.core.agent.knowledge.types import KnowledgeChunk, KnowledgeDocument


def chunk_document(
    document: KnowledgeDocument,
    *,
    chunker: Optional[HybridChunker] = None,
    converter: Optional[DocumentConverter] = None,
) -> list[KnowledgeChunk]:
    """Chunk `document` with Docling's hybrid (structure + token) chunker.

    `document.content` is treated as Markdown. Each returned chunk's
    ``content`` is the context-enriched serialization from
    :meth:`~docling.chunking.HybridChunker.contextualize`, which is what
    should be embedded.

    Args:
        document: Source document whose ``content`` is Markdown.
        chunker: Optional preconfigured :class:`~docling.chunking.HybridChunker`.
            Defaults to Docling's hybrid chunker.
        converter: Optional :class:`~docling.document_converter.DocumentConverter`.
            Defaults to a Markdown-only converter so PDF/OCR pipelines are
            not loaded.

    Returns:
        Chunks in document order. Empty when `document.content` is blank.
    """
    if not document.content.strip():
        return []

    converter = converter or DocumentConverter(allowed_formats=[InputFormat.MD])
    chunker = chunker or HybridChunker()
    conversion = converter.convert_string(
        document.content,
        format=InputFormat.MD,
        name=document.id,
    )

    chunks: list[KnowledgeChunk] = []
    for chunk in chunker.chunk(dl_doc=conversion.document):
        chunks.append(
            KnowledgeChunk(
                content=chunker.contextualize(chunk),
                source=document.url,
            )
        )
    return chunks


