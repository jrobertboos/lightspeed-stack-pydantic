from dataclasses import dataclass
from typing import NamedTuple, Optional

from pydantic_ai import AgentRunResult
from pydantic_ai.tools import ToolDefinition


class AgentTool(NamedTuple):
    """A tool definition paired with the label of the toolset it came from."""

    toolset: str
    definition: ToolDefinition


@dataclass(slots=True)
class AgentQuery:
    """Result of an agent query.

    For non-streaming queries, ``run_result`` is set immediately. For
    streaming queries it starts as ``None`` and is populated in place once
    the SSE stream has been fully consumed (staying ``None`` if the run
    errors out before completing), so the mutable dataclass -- rather than
    an immutable ``NamedTuple`` -- is required here.
    """

    run_result: Optional[AgentRunResult] = None
