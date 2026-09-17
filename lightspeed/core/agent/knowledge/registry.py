"""Process-wide registry mapping knowledge source names to :class:`VectorStore` instances.

There is no built-in :class:`~lightspeed.core.agent.knowledge.store.VectorStore`
backend for any :class:`~lightspeed.app.models.config.KnowledgeSourceType`
yet (see :mod:`lightspeed.core.agent.knowledge.store`), so
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

from lightspeed.core.agent.knowledge.store import VectorStore
from lightspeed.core.types import Singleton


class KnowledgeStoreRegistry(metaclass=Singleton):
    """Process-wide singleton holding hand-registered knowledge :class:`VectorStore` instances."""

    def __init__(self) -> None:
        self._stores: dict[str, VectorStore] = {}

    def register(self, name: str, store: VectorStore) -> None:
        """Register `store` under `name`, replacing any store already registered for it."""
        self._stores[name] = store

    def unregister(self, name: str) -> None:
        """Remove the store registered under `name`, if any. No-op if absent."""
        self._stores.pop(name, None)

    def get(self, name: str) -> VectorStore:
        """Return the store registered under `name`.

        Raises:
            KeyError: If no store is registered for `name`.
        """
        try:
            return self._stores[name]
        except KeyError as exc:
            raise KeyError(
                f"No VectorStore registered for knowledge source {name!r}. There is no built-in "
                "backend yet for any KnowledgeSourceType -- call "
                f"KnowledgeStoreRegistry().register({name!r}, store) with a VectorStore instance "
                "before building agents."
            ) from exc

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._stores

    def __len__(self) -> int:
        return len(self._stores)

    def __iter__(self) -> Iterator[str]:
        return iter(self._stores)

    @property
    def stores(self) -> Mapping[str, VectorStore]:
        """Read-only view of registered stores, keyed by source name."""
        return dict(self._stores)
