"""Load pydantic-ai agents from configured Lightspeed settings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.settings import ModelSettings

from lightspeed.src.providers.registry import ProviderRegistry


def load_agent(
    registry: ProviderRegistry,
    provider_name: str,
    model_name: str,
    *,
    instructions: Optional[str] = None,
    settings: Optional[ModelSettings] = None,
    capabilities: Optional[Sequence[AgentCapability[None]]] = None,
) -> Agent[None, str]:
    """Build a pydantic-ai :class:`~pydantic_ai.agent.Agent` from registry settings.

    Resolves a :class:`~pydantic_ai.models.Model` via
    :meth:`~lightspeed.src.providers.registry.ProviderRegistry.get_model`, then
    constructs an ``Agent`` ready for ``await agent.run(...)`` (or streaming).

    This is the Lightspeed rewrite counterpart to the original ``build_agent``
    helper, without Llama Stack.

    Parameters:
        registry: Registry of client-configured providers.
        provider_name: Registry key of the provider to use.
        model_name: Upstream model identifier (e.g. ``gpt-4o``).
        instructions: Optional system instructions for the agent.
        settings: Optional pydantic-ai model settings applied to the model.
        capabilities: Optional pydantic-ai capabilities (skills, etc.).

    Returns:
        An ``Agent`` bound to the configured provider and model.

    Raises:
        KeyError: If ``provider_name`` is not registered.
        ValueError: If ``model_name`` is empty.
    """
    model = registry.get_model(provider_name, model_name, settings=settings)
    return Agent(
        model,
        instructions=instructions,
        capabilities=capabilities,
    )
