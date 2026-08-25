"""Discover the tools registered on a pydantic-ai agent.

Tool discovery is intentionally decoupled from running the agent: each
toolset's ``get_tools`` is called directly, so this never invokes a tool
function or makes a model request. That makes it safe to call even for
tools with real side effects (file writes, external API calls, ...), and
cheap enough to call on every ``GET /tools`` request.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage


class AgentTool(NamedTuple):
    """A tool definition paired with the label of the toolset it came from."""

    toolset: str
    definition: ToolDefinition


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
    run_context: RunContext[Any] = RunContext(deps=deps, model=model, usage=RunUsage())

    tools: list[AgentTool] = []
    for toolset in agent.toolsets:
        toolset_tools = await toolset.get_tools(run_context)
        tools.extend(
            AgentTool(toolset=toolset.label, definition=tool.tool_def)
            for tool in toolset_tools.values()
        )
    return tools
