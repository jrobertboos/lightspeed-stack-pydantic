"""Build pydantic-ai agents from the process-wide :class:`ProviderRegistry`.

Providers and their available models are resolved once at startup by
:meth:`~lightspeed.core.providers.registry.ProviderRegistry.load`. Building an
``Agent`` here is just a lookup: no provider/model construction happens per
request.
"""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Agent

from lightspeed.core.providers.registry import ProviderRegistry


class AgentFactory:
    """Builds pydantic-ai :class:`~pydantic_ai.agent.Agent` instances on demand."""

    @staticmethod
    def create_agent(
        *,
        provider: str,
        model: str,
        instructions: Optional[str] = None,
        # no_tools / shield_ids / vector_store_ids will gate toolsets once
        # tools, safety, and RAG are wired into the rewrite.
    ) -> Agent[None, str]:
        """Build an ``Agent`` for one request.

        Parameters:
            provider: Name of the configured provider to use (e.g. from
                ``QueryRequest.provider``).
            model: Upstream model identifier (e.g. from ``QueryRequest.model``).
            instructions: Optional system instructions for the agent (e.g.
                from ``QueryRequest.system_prompt``).

        Returns:
            An ``Agent`` bound to the resolved provider and model.

        Raises:
            KeyError: If ``provider`` is not registered, or ``model`` is not
                available for ``provider``.
        """
        resolved_model = ProviderRegistry().get_model(provider, model)
        return Agent(resolved_model, instructions=instructions)
