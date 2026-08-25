"""Successful response models for the tools endpoint."""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, Field

from lightspeed.core.agent.tools import AgentTool


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

    @classmethod
    def from_agent_tool(cls, tool: AgentTool) -> Self:
        """Create a ToolInfo from an AgentTool.

        Args:
            tool: The pydantic-ai tool definition, paired with the label of
                the toolset it came from.

        Returns:
            A ToolInfo with the tool's identifier, description, toolset,
            and parameters (derived from its JSON Schema) populated.
        """
        schema = tool.definition.parameters_json_schema or {}
        required = set(schema.get("required", []))
        properties: dict[str, Any] = schema.get("properties", {})
        parameters = [
            ToolParameter(
                name=name,
                description=property_schema.get("description", ""),
                parameter_type=property_schema.get("type", "string"),
                required=name in required,
                default=property_schema.get("default"),
            )
            for name, property_schema in properties.items()
        ]
        return cls(
            identifier=tool.definition.name,
            description=tool.definition.description or "",
            parameters=parameters,
            toolset=tool.toolset,
        )


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

    @classmethod
    def from_agent_tools(cls, tools: list[AgentTool]) -> Self:
        """Create a ToolsResponse from a list of AgentTools.

        Args:
            tools: The list of pydantic-ai tool definitions, paired with the
                label of the toolset they came from.

        Returns:
            A ToolsResponse with the tools' identifiers, descriptions, toolsets,
            and parameters (derived from their JSON Schemas) populated.
        """
        return cls(tools=[ToolInfo.from_agent_tool(tool) for tool in tools])