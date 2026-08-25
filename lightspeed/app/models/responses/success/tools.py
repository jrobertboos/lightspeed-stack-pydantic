"""Successful response models for the tools endpoint."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ToolParameter(BaseModel):
    """A single parameter accepted by a tool."""

    name: str = Field(
        ...,
        description="Parameter name.",
        examples=["city"],
    )

    description: str = Field(
        "",
        description="Parameter description.",
    )

    parameter_type: str = Field(
        "string",
        description="JSON Schema type of the parameter (string, integer, boolean, ...).",
    )

    required: bool = Field(
        False,
        description="Whether the parameter must be supplied when calling the tool.",
    )

    default: Optional[Any] = Field(
        None,
        description="Default value used when the parameter is omitted, if any.",
    )

    model_config = ConfigDict(extra="forbid")


class ToolInfo(BaseModel):
    """A single tool (or other callable capability) available to the agent.

    Minimal rewrite of the original Lightspeed ``CatalogTool``: pydantic-ai
    tools aren't registered against Llama Stack providers/toolgroups, so
    ``toolset`` names the pydantic-ai toolset (an MCP server, a built-in
    toolset, ...) a tool came from instead of ``provider_id``/``toolgroup_id``.
    """

    identifier: str = Field(
        ...,
        description="Unique tool identifier.",
        examples=["get_weather"],
    )

    description: str = Field(
        "",
        description="What the tool does.",
    )

    parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="Parameters accepted by the tool.",
    )

    toolset: str = Field(
        ...,
        description="Name of the toolset (MCP server, builtin, ...) the tool came from.",
        examples=["builtin", "mcp:filesystem"],
    )

    model_config = ConfigDict(extra="forbid")


class ToolsResponse(BaseModel):
    """Response body for ``GET /v1/tools``.

    Minimal rewrite shape aligned with the original Lightspeed
    ``ToolsResponse``. MCP servers, built-in toolgroups, and skills are not
    yet wired into :class:`~lightspeed.core.agent.factory.AgentFactory` (see
    ``mcp_servers`` / ``skills`` in
    :class:`~lightspeed.app.models.config.Configuration`), so ``tools`` is
    empty until they land.
    """

    tools: list[ToolInfo] = Field(
        default_factory=list,
        description="Tools and other callable capabilities available to the agent.",
        examples=[
            [
                {
                    "identifier": "get_weather",
                    "description": "Get the current weather for a city.",
                    "parameters": [
                        {
                            "name": "city",
                            "description": "The city to get the weather for.",
                            "parameter_type": "string",
                            "required": True,
                            "default": None,
                        }
                    ],
                    "toolset": "builtin",
                },
            ]
        ],
    )

    model_config = ConfigDict(extra="forbid")
