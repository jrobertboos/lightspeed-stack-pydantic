"""Successful response model for the root ("/") endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RootResponse(BaseModel):
    """Response body for ``GET /``.

    Minimal rewrite of the original static HTML index page: a small JSON
    banner pointing at the interactive API docs instead of rendered HTML.
    """

    name: str = Field(
        ...,
        description="Service name, from configuration.",
        examples=["Lightspeed Stack"],
    )

    docs: str = Field(
        "/docs",
        description="Path to the interactive Swagger UI documentation.",
    )

    model_config = ConfigDict(extra="forbid")
