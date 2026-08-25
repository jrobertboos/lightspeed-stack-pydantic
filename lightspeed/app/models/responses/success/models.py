"""Successful response models for the models endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelInfo(BaseModel):
    """A single model available through a configured provider.

    Minimal rewrite of the original Lightspeed ``CatalogModel``: Llama Stack
    catalog fields (``metadata``, ``api_model_type``, ``model_type``, ...)
    don't apply to pydantic-ai providers and are dropped.
    """

    identifier: str = Field(
        ...,
        description="Fully qualified model identifier, as 'provider/model'.",
        examples=["openai/gpt-4-turbo"],
    )

    provider_name: str = Field(
        ...,
        description="Name of the provider (registry entry) this model belongs to.",
        examples=["openai"],
    )

    model_name: str = Field(
        ...,
        description="Model name as reported by the provider.",
        examples=["gpt-4-turbo"],
    )

    model_config = ConfigDict(extra="forbid")


class ModelsResponse(BaseModel):
    """Response body for ``GET /v1/models``.

    Minimal rewrite shape aligned with the original Lightspeed
    ``ModelsResponse``. The original ``model_type`` query filter is not yet
    implemented.
    """

    models: list[ModelInfo] = Field(
        ...,
        description="List of models available across all configured providers.",
        examples=[
            [
                {
                    "identifier": "openai/gpt-4-turbo",
                    "provider_name": "openai",
                    "model_name": "gpt-4-turbo",
                },
            ]
        ],
    )

    model_config = ConfigDict(extra="forbid")
