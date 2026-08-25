"""Handler for REST API call to list tools available to the agent.

Minimal rewrite of the original Lightspeed ``/tools`` endpoint. The original
consolidates tools from configured MCP servers plus built-in toolgroups
(file search, agent "skills" capabilities). This lists whatever toolsets
:meth:`~lightspeed.core.agent.factory.AgentFactory.create_agent` attaches to
its agents -- currently none, since MCP servers, built-in toolgroups, and
skills aren't wired in yet (see the ``mcp_servers`` / ``skills`` stub fields
on :class:`~lightspeed.app.models.config.Configuration`). Auth/authorization
and MCP headers are not yet implemented either.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from lightspeed.app.models.responses.success.tools import (
    ToolInfo,
    ToolParameter,
    ToolsResponse,
)
from lightspeed.core.agent.factory import AgentFactory
from lightspeed.core.agent.tools import AgentTool, list_agent_tools

router = APIRouter(tags=["tools"])


@router.get("/tools", summary="Tools Endpoint Handler")
async def tools_endpoint_handler() -> ToolsResponse:
    """Return the tools and other callable capabilities available to the agent.

    Builds an agent via ``AgentFactory.create_agent`` with no provider/model
    (tool discovery never calls the model or a tool, so the placeholder
    model that gets bound instead is harmless) so this always has the same
    toolsets a real request agent would, reflecting reality as they're
    wired in.
    """
    agent = AgentFactory.create_agent()
    agent_tools = await list_agent_tools(agent)
    return ToolsResponse(tools=[_tool_info(tool) for tool in agent_tools])


def _tool_info(tool: AgentTool) -> ToolInfo:
    """Map a pydantic-ai tool definition onto the ``/tools`` response shape."""
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
    return ToolInfo(
        identifier=tool.definition.name,
        description=tool.definition.description or "",
        parameters=parameters,
        toolset=tool.toolset,
    )
