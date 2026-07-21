"""Successful response models for the query endpoint."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class QueryResponse(BaseModel):
    """Response body for ``POST /v1/query``.

    Minimal rewrite shape aligned with the original Lightspeed ``QueryResponse``.
    Tool/RAG/quota fields default empty until those flows are implemented.
    """

    conversation_id: Optional[str] = Field(
        None,
        description="The optional conversation ID (UUID)",
        examples=["c5260aec-4d82-4370-9fdf-05cf908b3f16"],
    )

    response: str = Field(
        ...,
        description="Response from LLM",
        examples=[
            "Kubernetes is an open-source container orchestration system for automating ..."
        ],
    )

    # Parity fields — not yet populated by the rewrite endpoint.
    rag_chunks: list[Any] = Field(default_factory=list)
    referenced_documents: list[Any] = Field(default_factory=list)
    truncated: bool = Field(False)
    input_tokens: int = Field(0)
    output_tokens: int = Field(0)
    available_quotas: dict[str, int] = Field(default_factory=dict)
    tool_calls: list[Any] = Field(default_factory=list)
    tool_results: list[Any] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
