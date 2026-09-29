"""``Knowledge`` capability: vector-similarity search ("RAG") over a pluggable ``VectorStore``.

Exposed to the agent as a model-callable tool (`mode='tool'`), as automatic
inline context injection (`mode='inline'`), or both by attaching two
instances -- configure sources under `knowledge.strategy.tool` / `.inline` in
`lightspeed-stack.yaml` (see
:class:`~lightspeed.core.agent.knowledge.factory.KnowledgeCapabilityFactory`),
or construct this directly for programmatic use, the same way
:class:`pydantic_ai_harness.memory.Memory` is used standalone.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Set

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import TextContent, UserContent
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from lightspeed.core.agent.knowledge.toolset import KnowledgeToolset
from lightspeed.core.agent.utils import append_latest_message, extract_latest_message_text



class KnowledgeSource(ABC):
    """Knowledge source to search."""

    name: str
    """Name of the knowledge source."""

    @abstractmethod
    async def search(self, prompt: str) -> list[KnowledgeMatch]:
        """Search the knowledge source for matches."""


@dataclass(frozen=True)
class KnowledgeMatch:
    """One scored knowledge-search result."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Unique identifier for this match."""

    content: str
    """The matched chunk's text."""

    score: float
    """Similarity score for this match; higher is more relevant."""

    source: str | None = None
    """Optional citation identifier (e.g. a document URI), for grounding."""

    metadata: Mapping[str, Any] = field(default_factory=dict)
    """Optional backend-specific metadata carried alongside the match."""


@dataclass
class Knowledge(AbstractCapability[AgentDepsT]):
    """Vector-similarity search ("RAG") over one named :class:`KnowledgeSource`."""

    source: KnowledgeSource
    """Knowledge source to search."""

    mode: Set[Literal["tool", "auto"]] = {"tool"}
    """
    `'tool'`: exposes a model-callable `search_<name>` tool.
    `'auto'`: searches the latest user prompt automatically and injects
    matches as bounded, delimited context on every model request.
    """

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `search_<name>` toolset, only in `'tool'` mode."""
        return KnowledgeToolset(self) if self.mode == "tool" else None

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """In `'auto'` mode, search the latest user prompt and inject matches as context."""

        def to_message(matches: list[KnowledgeMatch], source: str) -> Optional[UserContent]:
            if not matches:
                return None

            match_blocks = [
                f'<match id="{match.id}" source="{match.source}">\n'
                f"{match.content.strip()}\n"
                f"</match>"
                for match in matches
            ]
            
            body = "\n\n".join(match_blocks)

            return TextContent(
                content=f'<knowledge source="{source}">\n{body}\n</knowledge>',
                metadata={
                    "kind": "knowledge",
                    "knowledge_source": source,
                    "knowledge_scores": {match.id: match.score for match in matches},
                    "knowledge_metadata": {match.id: match.metadata for match in matches},
                },
            )

        if "auto" in self.mode: 

            prompt = extract_latest_message_text(request_context.messages)
            if not prompt:
                return request_context

            matches = await self.source.search(prompt)

            if message := to_message(matches, source=self.name):
                append_latest_message(request_context.messages, message)

        return request_context