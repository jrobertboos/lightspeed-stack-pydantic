"""Discover the tools registered on a pydantic-ai agent.

Tool discovery is intentionally decoupled from running the agent: each
toolset's ``get_tools`` is called directly, so this never invokes a tool
function or makes a model request. That makes it safe to call even for
tools with real side effects (file writes, external API calls, ...), and
cheap enough to call on every ``GET /tools`` request.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from lightspeed.core.agent.schemas import AgentTool


def _build_capabilities(agent: Agent[Any, Any]) -> dict[str, AbstractCapability[Any]]:
    """Register every capability on ``agent`` under some unique key.

    Toolsets contributed by a capability (e.g. MCP) are wrapped in a
    ``CapabilityOwnedToolset`` that looks itself up in ``RunContext.capabilities``
    by identity (not by key) when ``get_tools`` is called -- a real agent run
    populates that registry as part of building its ``RunContext``, but a
    synthetic, discovery-only ``RunContext`` has no run to inherit it from.
    The key itself only needs to be unique here (an explicit ``id``, else its
    position), not to match what a real run would assign: nothing reads it
    back except that identity scan.
    """
    capabilities: list[AbstractCapability[Any]] = []
    agent.root_capability.apply(capabilities.append)
    return {
        capability.id or f"_capability_{index}": capability
        for index, capability in enumerate(capabilities)
    }


async def list_agent_tools(agent: Agent[Any, Any], *, deps: Any = None) -> list[AgentTool]:
    """Return every tool exposed by ``agent``'s registered toolsets.

    Iterates :attr:`Agent.toolsets` and calls each toolset's ``get_tools``
    directly (the same call the agent graph makes to build the tool schemas
    sent to the model) rather than running the agent, so no tool is ever
    called and no model request is made.

    Parameters:
        agent: The agent to discover tools on.
        deps: Dependencies passed through to each toolset's ``get_tools``,
            for toolsets that filter or prepare tools based on ``ctx.deps``.
            Defaults to None, matching agents with no deps type (e.g.
            ``AgentFactory.create_agent``'s ``Agent[None, str]``).

    Returns:
        One :class:`AgentTool` per tool, tagged with the label of the
        toolset it came from.
    """
    # get_tools only needs *some* Model to satisfy RunContext -- it never
    # calls it -- so agents built purely for discovery (no bound model) get
    # a harmless placeholder instead of requiring a real one.
    model = agent.model if agent.model is not None else TestModel()
    run_context: RunContext[Any] = RunContext(
        deps=deps, model=model, usage=RunUsage(), capabilities=_build_capabilities(agent)
    )

    tools: list[AgentTool] = []
    for toolset in agent.toolsets:
        toolset_tools = await toolset.get_tools(run_context)
        # Each `ToolsetTool.toolset` is the toolset that actually defined it (e.g.
        # the `MCPToolset` itself), not `toolset` above: wrappers like
        # `CombinedToolset` and `CapabilityOwnedToolset` pass it through
        # unchanged rather than overwriting it with themselves, so using it
        # here avoids labels like "CombinedToolset(CapabilityOwnedToolset(...))".
        # Prefer `id` (e.g. "mcp:test-mcp") over the more verbose `label`
        # (e.g. "MCPToolset 'mcp:test-mcp'"), falling back to `label` only for
        # toolsets with no `id` (e.g. tools registered directly on the agent).
        tools.extend(
            AgentTool(toolset=tool.toolset.id or tool.toolset.label, definition=tool.tool_def)
            for tool in toolset_tools.values()
        )
    return tools
