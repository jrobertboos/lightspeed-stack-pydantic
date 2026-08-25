"""Successful response models for the health probe endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LivenessResponse(BaseModel):
    """Response body for ``GET /liveness``."""

    alive: bool = Field(
        ...,
        description="Flag indicating that the app is alive.",
        examples=[True],
    )

    model_config = ConfigDict(extra="forbid")


class ReadinessResponse(BaseModel):
    """Response body for ``GET /readiness``.

    Minimal rewrite: readiness reflects whether providers were loaded at
    startup. Per-provider health checks (as in the original) are not yet
    implemented.
    """

    ready: bool = Field(
        ...,
        description="Flag indicating if the service is ready to handle requests.",
        examples=[True],
    )

    reason: str = Field(
        ...,
        description="The reason for the readiness status.",
        examples=["All configured providers are loaded"],
    )

    model_config = ConfigDict(extra="forbid")
