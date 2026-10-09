from dataclasses import dataclass, field

from lightspeed.core.agent.knowledge.sources.base import KnowledgeSource
from lightspeed.core.agent.knowledge.types import KnowledgeMatch


@dataclass
class OkpKnowledgeSource(KnowledgeSource):
    """Knowledge source backed by OKP.

    Unlike Faiss and pgvector, OKP is not an embedding index: this type
    subclasses :class:`KnowledgeSource` directly and implements ``search``
    against the OKP MCP API. Not implemented yet — register a built instance via
    :class:`~lightspeed.core.agent.knowledge.registry.KnowledgeSourceRegistry`
    once a client exists.
    """

    url: str = field(kw_only=True)
    """URL of the OKP MCP server."""

    async def search(self, prompt: str) -> list[KnowledgeMatch]:
        """Search OKP for matches relevant to `prompt`."""
        raise NotImplementedError("OkpKnowledgeSource.search is not implemented yet")
