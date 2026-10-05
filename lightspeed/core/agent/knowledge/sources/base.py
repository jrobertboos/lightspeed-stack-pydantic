from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Set
import uuid

@dataclass(frozen=True)
class KnowledgeMatch:
    """One scored knowledge-search result."""

    content: str
    """The matched chunk's text."""

    score: float
    """Similarity score for this match; higher is more relevant."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this match."""

    source: str | None = None
    """Optional citation identifier (e.g. a document URI), for grounding."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Optional backend-specific metadata carried alongside the match."""


@dataclass
class KnowledgeSource(ABC):
    """Knowledge source to search, with its own exposure mode.

    `mode` lives here (rather than on whatever capability attaches this
    source) so each source controls how *it* is exposed, independent of how
    many other sources a `Knowledge` capability combines it with.
    """

    name: str
    """Name of the knowledge source."""

    mode: Set[Literal["tool", "auto"]] = field(default_factory=lambda: {"tool"}, kw_only=True)
    """
    `'tool'`: exposes a model-callable `search_<name>` tool for this source.
    `'auto'`: searches the latest user prompt automatically and injects this
    source's matches as bounded, delimited context on every model request.
    """

    @abstractmethod
    async def search(self, prompt: str) -> list[KnowledgeMatch]:
        """Search the knowledge source for matches."""
