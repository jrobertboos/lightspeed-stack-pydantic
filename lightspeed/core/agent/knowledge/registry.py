"""Process-wide registries for the `Knowledge` capability's lazily-built, hand-registered dependencies.

:class:`KnowledgeSourceRegistry` maps knowledge source names to
:class:`VectorStore` instances. There is no built-in
:class:`~lightspeed.core.agent.knowledge.store.VectorStore` backend for any
:class:`~lightspeed.app.models.config.KnowledgeSourceType` yet (see
:mod:`lightspeed.core.agent.knowledge.store`), so
:class:`~lightspeed.core.agent.knowledge.factory.KnowledgeCapabilityFactory`
can't build one from a
:class:`~lightspeed.app.models.config.KnowledgeSourceConfiguration` the way
:class:`~lightspeed.core.providers.registry.ProviderRegistry` builds a
pydantic-ai ``Provider`` from a ``ProviderConfiguration``. Registering a store
here by name -- e.g. at application startup, alongside
``ProviderRegistry().load(...)`` -- is how a configured source becomes
usable until a concrete backend builder exists.

:class:`CrossEncoderRegistry` caches lazily-loaded `sentence-transformers`
`CrossEncoder` models by name, for use by
:mod:`~lightspeed.core.agent.knowledge.reranker`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterator, Mapping

from lightspeed.core.agent.knowledge.sources.base import KnowledgeSource
from lightspeed.core.types import Singleton
from sentence_transformers import CrossEncoder

class KnowledgeSourceRegistry(metaclass=Singleton):
    """Process-wide singleton holding hand-registered knowledge :class:`KnowledgeSource` instances."""

    def __init__(self) -> None:
        self._sources: dict[str, KnowledgeSource] = {}

    def register(self, name: str, source: KnowledgeSource) -> None:
        """Register `source` under `name`, replacing any source already registered for it."""
        self._sources[name] = source

    def unregister(self, name: str) -> None:
        """Remove the source registered under `name`, if any. No-op if absent."""
        self._sources.pop(name, None)

    def get(self, name: str) -> KnowledgeSource:
        """Return the source registered under `name`.

        Raises:
            KeyError: If no source is registered for `name`.
        """
        try:
            return self._sources[name]
        except KeyError as exc:
            raise KeyError(
                f"No KnowledgeSource registered for knowledge source {name!r}. There is no built-in "
                "backend yet for any KnowledgeSourceType -- call "
                f"KnowledgeSourceRegistry().register({name!r}, source) with a KnowledgeSource instance "
                "before building agents."
            ) from exc

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._sources

    def __len__(self) -> int:
        return len(self._sources)

    def __iter__(self) -> Iterator[str]:
        return iter(self._sources)

    @property
    def sources(self) -> Mapping[str, KnowledgeSource]:
        """Read-only view of registered sources, keyed by source name."""
        return dict(self._sources)


class CrossEncoderRegistry(metaclass=Singleton):
    """Process-wide singleton caching lazily-loaded `sentence-transformers` `CrossEncoder` models by name.

    Used by :mod:`~lightspeed.core.agent.knowledge.reranker` so a reranker
    model is loaded (an expensive, blocking operation) at most once per
    process, however many times it's referenced by name.
    """

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, model_name: str) -> Any:
        """Return the `CrossEncoder` for `model_name`, loading and caching it on first use."""
        if model_name not in self._models:
            async with self._lock:
                    self._models[model_name] = await asyncio.to_thread(CrossEncoder, model_name)
        return self._models[model_name]
