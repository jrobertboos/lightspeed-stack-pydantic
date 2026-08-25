"""Build pydantic-ai MCP capabilities from configured MCP servers.

Each configured server is wrapped in a :class:`~pydantic_ai.capabilities.MCP`
capability -- described by pydantic-ai as "the primary entry point for using
MCP servers with Pydantic AI" -- rather than attaching a bare
:class:`~pydantic_ai.mcp.MCPToolset` directly via ``toolsets=``. A
capability's contributed toolset still flows into ``Agent.toolsets`` (see
:mod:`lightspeed.core.agent.tools.discovery`), so tool discovery needs no
special-casing, while leaving room for a later ``native=True`` upgrade
(provider-native MCP) without changing the config schema.
"""

from __future__ import annotations

from typing import Any, Iterable

from pydantic_ai.capabilities import MCP, AgentCapability
from pydantic_ai.mcp import MCPToolset

from lightspeed.app.models.config import MCPServerConfiguration


class MCPCapabilityFactory:
    """Builds :class:`~pydantic_ai.capabilities.MCP` capabilities from configured MCP servers."""

    @staticmethod
    def build_capabilities(
        configs: Iterable[MCPServerConfiguration],
    ) -> list[AgentCapability[Any]]:
        """Build one ``MCP`` capability per configured server.

        Parameters:
            configs: The configured MCP servers (e.g.
                ``Configuration.mcp_servers``).

        Returns:
            One ``MCP`` capability per config, suitable for
            ``Agent(capabilities=...)``.

        Raises:
            ValueError: If two configs share a name.
            NotImplementedError: If a config sets ``forward_headers``, which
                needs per-request context not wired up yet (see
                :meth:`_build_capability`).
        """
        seen: set[str] = set()
        capabilities: list[AgentCapability[Any]] = []
        for config in configs:
            if config.name in seen:
                raise ValueError(f"MCP server already registered: {config.name!r}")
            seen.add(config.name)
            capabilities.append(_build_capability(config))
        return capabilities


def _build_capability(config: MCPServerConfiguration) -> MCP[Any]:
    """Build a single ``MCP`` capability, with a fully-configured local ``MCPToolset``."""
    if config.forward_headers:
        raise NotImplementedError(
            f"MCP server {config.name!r} sets forward_headers, which needs "
            "per-request context (the incoming request's headers) that "
            "isn't wired into the agent factory yet."
        )

    toolset_id = f"mcp:{config.name}"
    url = str(config.url)
    read_timeout = float(config.timeout) if config.timeout else None

    toolset = MCPToolset(
        url,
        id=toolset_id,
        include_instructions=True,
        read_timeout=read_timeout,
    )
    return MCP(url, native=False, local=toolset, id=toolset_id)
