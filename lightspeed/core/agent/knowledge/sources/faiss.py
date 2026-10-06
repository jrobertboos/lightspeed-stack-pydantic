import asyncio
import base64
import io
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Optional, Sequence, Set

import faiss
import numpy as np
from pydantic_ai.embeddings import Embedder, EmbeddingSettings

from lightspeed.core.agent.knowledge.sources.base import KnowledgeMatch
from lightspeed.core.agent.knowledge.sources.vector_store import VectorStore, VectorStoreKnowledgeSource

_KV_NAMESPACE: Final[str] = "vector_io::faiss"
_KV_VERSION: Final[str] = "v3"


@dataclass(frozen=True)
class FaissRecord:
    """Text and metadata for one vector stored in a :class:`FaissVectorStore`, keyed by its row in the index."""

    content: str
    """The chunk of text this vector represents."""

    source: str | None = None
    """Optional citation identifier (e.g. a document URI), for grounding."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Optional backend-specific metadata to carry alongside a match."""


class FaissVectorStore(VectorStore):
    """Vector store backed by a local sqlite-faiss BYOK database file.

    Reads the sqlite-faiss kvstore layout that ``rag-content`` writes: a
    SQLite ``kvstore`` table keyed by
    ``vector_io::faiss:faiss_index:v3::<vector_store_id>``, whose value is a
    JSON blob holding a base64-encoded, serialized `faiss.Index` plus a
    ``chunk_by_index`` map from Faiss row id to chunk text/metadata. The
    database is read-only at search time, so the index and chunk map are
    loaded once, eagerly, in the constructor.

    A single file can hold more than one vector store; `vector_store_id`
    picks which one this instance serves.
    """

    def __init__(self, db_path: str | Path, vector_store_id: str) -> None:
        """Load the Faiss index and chunk map for `vector_store_id` from the sqlite-faiss file at `db_path`.

        Args:
            db_path: Path to the sqlite-faiss kvstore file.
            vector_store_id: Store id inside the file to load.

        Raises:
            KeyError: If `vector_store_id` isn't present in `db_path`.
        """
        self._db_path = str(db_path)
        self._vector_store_id = vector_store_id
        self._index, self._records = _load_store(self._db_path, vector_store_id)

    async def search(
        self, embeddings: Sequence[Sequence[float]], limit: int = 5, threshold: float = 0.0
    ) -> list[KnowledgeMatch]:
        """Search the vector store for matches.

        `embeddings` may contain more than one query vector (e.g. one per
        chunk of a multi-part query); matches for each are merged, deduped by
        index row (keeping the best score per row), and ranked together
        before being truncated to `limit`. Faiss L2 distances are converted
        to a bounded, higher-is-better score via ``1 / (1 + distance)``;
        rows scoring below `threshold` are dropped.
        """
        queries = np.asarray(embeddings, dtype=np.float32)
        # Faiss is a blocking C++ call; run it off the event loop thread.
        distances, indices = await asyncio.to_thread(self._index.search, queries, limit)

        best_score: dict[int, float] = {}
        for row_distances, row_indices in zip(distances, indices):
            for distance, idx in zip(row_distances, row_indices):
                if idx < 0 or idx not in self._records:
                    continue
                score = 1.0 / (1.0 + float(distance))
                if score < threshold:
                    continue
                if idx not in best_score or score > best_score[idx]:
                    best_score[idx] = score

        ranked = sorted(best_score.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            KnowledgeMatch(
                content=self._records[idx].content,
                score=score,
                source=self._records[idx].source,
                metadata=self._records[idx].metadata,
            )
            for idx, score in ranked
        ]


def _kv_key(vector_store_id: str) -> str:
    """Build the sqlite-faiss kvstore key for `vector_store_id`."""
    return f"{_KV_NAMESPACE}:faiss_index:{_KV_VERSION}::{vector_store_id}"


def _deserialize_index(payload: str) -> faiss.Index:
    """Decode a sqlite-faiss-stored Faiss index blob (base64 of an .npy uint8 array)."""
    raw = base64.b64decode(payload)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    return faiss.deserialize_index(arr)


def _load_store(db_path: str, vector_store_id: str) -> tuple[faiss.Index, dict[int, FaissRecord]]:
    """Read and deserialize a Faiss index plus its chunk records from a sqlite-faiss kvstore file."""
    connection = sqlite3.connect(db_path)
    try:
        row = connection.execute(
            "SELECT value FROM kvstore WHERE key = ?",
            (_kv_key(vector_store_id),),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise KeyError(f"No FAISS index found for vector_store_id={vector_store_id!r} in {db_path!r}")

    payload = json.loads(row[0])
    index = _deserialize_index(payload["faiss_index"])

    records: dict[int, FaissRecord] = {}
    for idx_str, chunk_json in payload["chunk_by_index"].items():
        chunk_data = json.loads(chunk_json)
        records[int(idx_str)] = FaissRecord(
            content=chunk_data["content"],
            source=chunk_data.get("chunk_id"),
            metadata=chunk_data.get("metadata", {}),
        )

    return index, records

class FaissKnowledgeSource(VectorStoreKnowledgeSource):
    """Knowledge source that uses Faiss for vector storage.

    Convenience wrapper around :class:`VectorStoreKnowledgeSource` that
    builds its :class:`FaissVectorStore` directly from a sqlite-faiss file,
    so callers don't need to construct the store separately.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        name: str,
        db_path: str | Path,
        vector_store_id: str,
        embedder: Embedder,
        embedding_settings: Optional[EmbeddingSettings] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        mode: Optional[Set[Literal["tool", "auto"]]] = None,
    ) -> None:
        """Build a Faiss-backed knowledge source from a sqlite-faiss file.

        Args:
            name: Name of this knowledge source.
            db_path: Path to the sqlite-faiss kvstore file.
            vector_store_id: Store id inside `db_path` to load.
            embedder: Embedder used to embed search queries.
            embedding_settings: Optional settings to use for the embedding.
            top_k: Maximum number of matches to return.
            score_threshold: Minimum similarity score to return.
            mode: How this source is exposed -- `'tool'`, `'auto'`, or both.
                Defaults to `{'tool'}`.
        """
        super().__init__(
            name=name,
            store=FaissVectorStore(db_path, vector_store_id),
            embedder=embedder,
            embedding_settings=embedding_settings,
            top_k=top_k,
            score_threshold=score_threshold,
            mode=mode if mode is not None else {"tool"},
        )
