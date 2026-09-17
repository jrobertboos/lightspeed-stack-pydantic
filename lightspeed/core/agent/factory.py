"""Build pydantic-ai agents from the process-wide :class:`ProviderRegistry`.

Providers and their available models are resolved once at startup by
:meth:`~lightspeed.core.providers.registry.ProviderRegistry.load`. Building an
``Agent`` here is just a lookup: no provider/model construction happens per
request.
"""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from lightspeed.core.agent.knowledge.factory import KnowledgeCapabilityFactory
from lightspeed.core.agent.tools.mcp.factory import MCPCapabilityFactory
from lightspeed.core.agent.tools.skills.factory import SkillsCapabilityFactory
from lightspeed.core.config import configuration
from lightspeed.core.providers.registry import ProviderRegistry


class AgentFactory:
    """Builds pydantic-ai :class:`~pydantic_ai.agent.Agent` instances on demand."""

    @staticmethod
    def create_agent(
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        instructions: Optional[str] = None,
        # no_tools / shield_ids / vector_store_ids will gate toolsets once
        # safety and RAG are wired into the rewrite.
    ) -> Agent[None, str]:
        """Build an ``Agent``, bound to a real provider/model or (when both
        are omitted) a throwaway agent for introspection only.

        The ``/tools`` endpoint calls this with no ``provider``/``model`` to
        get an agent with the same toolsets a real request agent would have,
        purely to list them -- it never calls the model or runs a tool.

        Every configured MCP server is attached as an ``MCP`` capability (see
        :class:`~lightspeed.core.agent.tools.mcp.factory.MCPCapabilityFactory`),
        configured skill paths as a ``Skills`` capability (see
        :class:`~lightspeed.core.agent.tools.skills.factory.SkillsCapabilityFactory`),
        and configured knowledge sources as ``Knowledge`` capabilities (see
        :class:`~lightspeed.core.agent.knowledge.factory.KnowledgeCapabilityFactory`),
        so ``Configuration`` must already be loaded (the ``/tools`` and
        ``/query`` endpoints both check this before calling here).

        Parameters:
            provider: Name of the configured provider to use (e.g. from
                ``QueryRequest.provider``). Omit together with ``model`` to
                get a harmless placeholder model instead of a real one.
            model: Upstream model identifier (e.g. from ``QueryRequest.model``).
            instructions: Optional system instructions for the agent (e.g.
                from ``QueryRequest.system_prompt``).

        Returns:
            An ``Agent`` bound to the resolved provider and model, or to a
            placeholder model if both ``provider`` and ``model`` are omitted.

        Raises:
            RuntimeError: If ``Configuration`` hasn't been loaded yet.
            KeyError: If ``provider`` is not registered, or ``model`` is not
                available for ``provider``.
        """
        if provider is None and model is None:
            resolved_model = TestModel()
        else:
            resolved_model = ProviderRegistry().get_model(provider, model)

        capabilities = MCPCapabilityFactory.build_capabilities(
            configuration.configuration.mcp_servers
        )
        capabilities.extend(
            SkillsCapabilityFactory.build_capabilities(configuration.configuration.skills)
        )
        capabilities.extend(
            KnowledgeCapabilityFactory.build_capabilities(configuration.configuration.knowledge)
        )
        return Agent(resolved_model, instructions=instructions, capabilities=capabilities)
