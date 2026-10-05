"""Build a ``Knowledge`` capability from configured knowledge sources.

All configured sources are combined onto a single
:class:`~lightspeed.core.agent.knowledge.capability.Knowledge` capability;
each source's own `mode` (see
:class:`~lightspeed.core.agent.knowledge.sources.base.KnowledgeSource`)
controls how *that* source is exposed -- as a model-callable tool, automatic
inline context injection, or both. Building a source's backend from
`type`/`config`/`embedding_model` is out of scope here: since no concrete
backend exists for any :class:`~lightspeed.app.models.config.KnowledgeSourceType`
yet, a source must already be registered -- with its own `mode` set -- in
:class:`~lightspeed.core.agent.knowledge.registry.KnowledgeSourceRegistry`.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic_ai.capabilities import AgentCapability

from lightspeed.app.models.config import KnowledgeConfiguration
from lightspeed.core.agent.capability_factory import CapabilityFactory
from lightspeed.core.agent.knowledge.capability import Knowledge
from lightspeed.core.agent.knowledge.registry import KnowledgeSourceRegistry


class KnowledgeCapabilityFactory(CapabilityFactory[Optional[KnowledgeConfiguration]]):
    """Builds a single :class:`~lightspeed.core.agent.knowledge.capability.Knowledge` capability."""

    @staticmethod
    def build_capabilities(config: Optional[KnowledgeConfiguration]) -> list[AgentCapability[Any]]:
        """Build one ``Knowledge`` capability wrapping every configured source.

        Parameters:
            config: The configured knowledge section (e.g.
                ``Configuration.knowledge``), or ``None`` when knowledge is
                omitted.

        Returns:
            A single-element list holding one ``Knowledge`` capability over
            all configured sources, suitable for ``Agent(capabilities=...)``.
            Empty when `config` is `None` or has no sources.

        Raises:
            ValueError: If two sources share a name.
            NotImplementedError: If ``knowledge.reranker`` is set (not yet
                implemented).
            KeyError: If a configured source has no
                :class:`~lightspeed.core.agent.knowledge.sources.base.KnowledgeSource`
                registered for it in
                :class:`~lightspeed.core.agent.knowledge.registry.KnowledgeSourceRegistry`.
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

        registry = KnowledgeSourceRegistry()
        sources = [registry.get(source.name) for source in config.sources]
        return [Knowledge(sources=sources, id="knowledge")]
