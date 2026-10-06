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
"""

from __future__ import annotations

from typing import Iterator, Mapping

from lightspeed.core.agent.knowledge.sources.base import KnowledgeSource
from lightspeed.core.types import Singleton

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
