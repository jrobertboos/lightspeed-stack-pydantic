"""Handler for REST API call to list tools available to the agent.

Minimal rewrite of the original Lightspeed ``/tools`` endpoint. The original
consolidates tools from configured MCP servers plus built-in toolgroups
(file search, agent "skills" capabilities). This lists whatever toolsets
:func:`~lightspeed.core.agent.service.list_tools` discovers on a freshly
built agent -- currently none, since MCP servers, built-in toolgroups, and
skills aren't wired in yet (see the ``mcp_servers`` / ``skills`` stub fields
on :class:`~lightspeed.app.models.config.Configuration`). Auth/authorization
and MCP headers are not yet implemented either.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from lightspeed.app.models.responses.error import InternalServerErrorResponse
from lightspeed.app.models.responses.success.tools import ToolInfo, ToolsResponse
from lightspeed.core import service as core_service
from lightspeed.core.agent import service as agent_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tools"])

tools_response: dict[int | str, dict[str, Any]] = {
    500: InternalServerErrorResponse.openapi_response(
        examples=["configuration", "tools"]
    ),
}


@router.get("/tools", responses=tools_response, summary="Tools Endpoint Handler")
async def tools_endpoint_handler() -> ToolsResponse:
    """Return the tools and other callable capabilities available to the agent.

    Delegates to ``agent_service.list_tools``, which reflects whatever
    toolsets a real request agent would have -- currently none, since MCP
    servers, built-in toolgroups, and skills aren't wired in yet.
    """
    try:
        core_service.get_configuration()
    except RuntimeError as exc:
        error_response = InternalServerErrorResponse.configuration_not_loaded()
        raise HTTPException(**error_response.model_dump()) from exc

    try:
        agent_tools = await agent_service.list_tools()
    except Exception as exc:
        logger.exception("Failed to list tools")
        error_response = InternalServerErrorResponse.tools_failed()
        raise HTTPException(**error_response.model_dump()) from exc

    return ToolsResponse.from_agent_tools(agent_tools)
