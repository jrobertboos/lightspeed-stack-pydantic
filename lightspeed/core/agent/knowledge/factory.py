"""Build ``Knowledge`` capabilities from configured knowledge sources.

Each configured source becomes one ``Knowledge`` capability per exposure mode
assigned to it under ``knowledge.strategy`` (``'tool'``, ``'inline'``, or
both) -- ``'tool'`` by default when a source isn't listed under either.
Building a source's :class:`~lightspeed.core.agent.knowledge.store.VectorStore`
is out of scope here: since no concrete backend exists for any
:class:`~lightspeed.app.models.config.KnowledgeSourceType` yet, one must
already be registered in
:class:`~lightspeed.core.agent.knowledge.registry.KnowledgeStoreRegistry`.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.embeddings import Embedder, infer_embedding_model
from pydantic_ai.providers import Provider, infer_provider

from lightspeed.app.models.config import KnowledgeConfiguration, KnowledgeSourceConfiguration
from lightspeed.core.agent.capability_factory import CapabilityFactory
from lightspeed.core.agent.knowledge.capability import Knowledge
from lightspeed.core.agent.knowledge.registry import KnowledgeStoreRegistry
from lightspeed.core.providers.registry import ProviderRegistry


class KnowledgeCapabilityFactory(CapabilityFactory[Optional[KnowledgeConfiguration]]):
    """Builds :class:`~lightspeed.core.agent.knowledge.capability.Knowledge` capabilities."""

    @staticmethod
    def build_capabilities(config: Optional[KnowledgeConfiguration]) -> list[AgentCapability[Any]]:
        """Build one ``Knowledge`` capability per configured source and assigned exposure mode.

        Parameters:
            config: The configured knowledge section (e.g.
                ``Configuration.knowledge``), or ``None`` when knowledge is
                omitted.

        Returns:
            One ``Knowledge`` capability per `(source, mode)` pair, suitable
            for ``Agent(capabilities=...)``. Empty when `config` is `None` or
            has no sources.

        Raises:
            ValueError: If two sources share a name, or
                ``knowledge.strategy`` names a source that isn't configured.
            NotImplementedError: If ``knowledge.reranker`` is set (not yet
                implemented).
            KeyError: If a configured source has no
                :class:`~lightspeed.core.agent.knowledge.store.VectorStore`
                registered for it in
                :class:`~lightspeed.core.agent.knowledge.registry.KnowledgeStoreRegistry`.
        """
        if config is None or not config.sources:
            return []

        if config.reranker is not None:
            raise NotImplementedError(
                "knowledge.reranker is set, but reranking isn't implemented yet. Remove it from "
                "the configuration."
            )

        seen: set[str] = set()
        for source in config.sources:
            if source.name in seen:
                raise ValueError(f"Knowledge source already registered: {source.name!r}")
            seen.add(source.name)

        inline_names = set(config.strategy.inline) if config.strategy else set()
        tool_names = set(config.strategy.tool) if config.strategy else set()
        unknown = (inline_names | tool_names) - seen
        if unknown:
            raise ValueError(f"knowledge.strategy names unconfigured source(s): {sorted(unknown)}")

        capabilities: list[AgentCapability[Any]] = []
        for source in config.sources:
            modes = [mode for mode, names in (("tool", tool_names), ("inline", inline_names)) if source.name in names]
            for mode in modes or ["tool"]:
                capabilities.append(_build_capability(source, mode))
        return capabilities


def _build_capability(source: KnowledgeSourceConfiguration, mode: str) -> Knowledge[Any]:
    """Build a single ``Knowledge`` capability for `source` in `mode`."""
    store = KnowledgeStoreRegistry().get(source.name)
    # `Embedder` itself has no `provider_factory` hook, unlike `infer_embedding_model` --
    # resolve the model with it first, then wrap the resolved model.
    embedding_model = infer_embedding_model(source.embedding_model, provider_factory=_provider_factory)
    embedder = Embedder(embedding_model)
    return Knowledge(
        name=source.name,
        store=store,
        embedder=embedder,
        top_k=source.top_k,
        mode=mode,  # type: ignore[arg-type]
        id=f"knowledge:{source.name}:{mode}",
    )


def _provider_factory(name: str) -> Provider[Any]:
    """Resolve an embedding model's provider, reusing a configured Lightspeed provider by name.

    Falls back to pydantic-ai's default `infer_provider` (env-var based
    credentials) when `name` isn't a provider configured in
    `Configuration.providers`.
    """
    registry = ProviderRegistry()
    if name in registry:
        return registry.get(name)
    return infer_provider(name)
