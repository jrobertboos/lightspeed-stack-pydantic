"""Request models for the query endpoint."""

from __future__ import annotations

from typing import Optional, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QueryRequest(BaseModel):
    """Request body for ``POST /v1/query``.

    Minimal rewrite shape aligned with the original Lightspeed ``QueryRequest``.
    Fields beyond ``query`` / ``provider`` / ``model`` / ``system_prompt`` are
    accepted for API compatibility but not yet wired in the endpoint.
    """

    query: str = Field(
        ...,
        description="The query string",
        examples=["What is Kubernetes?"],
    )

    conversation_id: Optional[str] = Field(
        None,
        description="The optional conversation ID (UUID)",
        examples=["c5260aec-4d82-4370-9fdf-05cf908b3f16"],
    )

    provider: Optional[str] = Field(
        None,
        description="The optional provider registry name",
        examples=["openai", "my-vllm"],
    )

    model: Optional[str] = Field(
        None,
        description="The optional model",
        examples=["gpt-4o"],
    )

    system_prompt: Optional[str] = Field(
        None,
        description="The optional system prompt / agent instructions",
        examples=["You are OpenShift assistant."],
    )

    # Accepted for parity; not yet used by the rewrite endpoint.
    no_tools: Optional[bool] = Field(
        False,
        description="Whether to bypass all tools and MCP servers",
    )
    generate_topic_summary: Optional[bool] = Field(
        True,
        description="Whether to generate topic summary for new conversations",
    )
    vector_store_ids: Optional[list[str]] = Field(
        None,
        description="Optional list of vector store IDs for RAG (not yet implemented)",
    )
    shield_ids: Optional[list[str]] = Field(
        None,
        description="Optional list of safety shield IDs (not yet implemented)",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_provider_and_model(self) -> Self:
        """Ensure provider and model are specified together."""
        if self.model and not self.provider:
            raise ValueError("Provider must be specified if model is specified")
        if self.provider and not self.model:
            raise ValueError("Model must be specified if provider is specified")
        return self
