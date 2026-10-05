"""``Knowledge`` capability: vector-similarity search ("RAG") over one or more pluggable ``KnowledgeSource``s.

Each attached :class:`~lightspeed.core.agent.knowledge.sources.base.KnowledgeSource`
declares its own `mode` -- exposed to the agent via a single model-callable
`search_knowledge` tool spanning every tool-mode source (`'tool'`), as
automatic inline context injection per source (`'auto'`), or both -- so a
single `Knowledge` capability can combine several sources, each exposed
however it needs to be. Configure sources under `knowledge.sources` in
`lightspeed-stack.yaml` (see
:class:`~lightspeed.core.agent.knowledge.factory.KnowledgeCapabilityFactory`),
or construct this directly for programmatic use, the same way
:class:`pydantic_ai_harness.memory.Memory` is used standalone.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import TextContent, UserContent
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from lightspeed.core.agent.knowledge.sources.base import KnowledgeMatch, KnowledgeSource
from lightspeed.core.agent.knowledge.toolset import KnowledgeToolset
from lightspeed.core.agent.utils import append_latest_message, extract_latest_message_text


@dataclass
class Knowledge(AbstractCapability[AgentDepsT]):
    """Vector-similarity search ("RAG") over one or more :class:`KnowledgeSource` instances."""

    sources: Sequence[KnowledgeSource]
    """Knowledge sources to search. Each source's own `mode` controls how it's exposed."""

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide a single `search_knowledge` tool spanning every source whose `mode` includes `'tool'`."""
        tool_sources = [source for source in self.sources if "tool" in source.mode]
        return KnowledgeToolset(tool_sources) if tool_sources else None

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """For each source whose `mode` includes `'auto'`, search the latest prompt and inject matches."""

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

        auto_sources = [source for source in self.sources if "auto" in source.mode]
        if not auto_sources:
            return request_context

        prompt = extract_latest_message_text(request_context.messages)
        if not prompt:
            return request_context

        for source in auto_sources:
            matches = await source.search(prompt)
            if message := to_message(matches, source=source.name):
                append_latest_message(request_context.messages, message)

        return request_context
