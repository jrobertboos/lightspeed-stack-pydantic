from typing import AsyncIterator, Optional, Tuple

from lightspeed.app.models.responses.error import InternalServerErrorResponse
from lightspeed.app.models.responses.success.stream import EndStreamPayload, ErrorStreamPayload, StartStreamPayload
from lightspeed.core.agent import AgentFactory
from lightspeed.core.agent.schemas import AgentQuery
from lightspeed.core.agent.streaming import StreamState, dispatch_stream_event
from pydantic_ai import Agent, AgentRunResultEvent


def create_agent(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    instructions: Optional[str] = None,
) -> Agent[None, str]:
    return AgentFactory.create_agent(provider=provider, model=model, instructions=instructions)

async def query(
    agent: Agent[None, str],
    query: str,
    *,
    conversation_id: Optional[str] = None
) -> AgentQuery:
    run_result = await agent.run(query, conversation_id=conversation_id)
    return AgentQuery(run_result=run_result)

async def stream_query(
    agent: Agent[None, str],
    query: str,
    *,
    conversation_id: Optional[str] = None
) -> Tuple[AsyncIterator[str], AgentQuery]:
    """Start a streaming agent run and return its SSE stream plus the eventual result.

    ``stream_query`` is a plain coroutine, not an async generator: Python
    disallows returning a value from an async generator, so the only way to
    hand back both the token stream and the eventual :class:`AgentRunResult`
    is to return the stream alongside a mutable :class:`AgentQuery` that the
    stream populates as it runs. ``agent_query.run_result`` is only set once
    the stream has been fully consumed (and stays ``None`` if the run errors
    out before completing). Chunk/text bookkeeping needed to build SSE
    payloads lives in a separate, private :class:`StreamState` -- callers
    only ever see the :class:`AgentQuery` result.

    Args:
        agent: Agent to execute.
        query: The user query to run.
        conversation_id: Conversation identifier to associate with the run
            and report in the ``start`` event.

    Returns:
        A ``(stream, agent_query)`` tuple. ``stream`` yields serialized SSE
        event strings (``start``, ``token``, ``turn_complete``, then either
        ``end`` or ``error``); ``agent_query.run_result`` holds the final
        :class:`AgentRunResult` after ``stream`` is exhausted.
    """
    state = StreamState()
    agent_query = AgentQuery()

    async def _stream() -> AsyncIterator[str]:
        yield StartStreamPayload.create(conversation_id=conversation_id).serialize_json()

        try:
            async with agent.run_stream_events(
                query, conversation_id=conversation_id
            ) as stream:
                async for event in stream:
                    if isinstance(event, AgentRunResultEvent):
                        agent_query.run_result = event.result
                    if payload := dispatch_stream_event(event, state):
                        yield payload.serialize_json()
        except Exception:  # pylint: disable=broad-except
            error_response = InternalServerErrorResponse.query_failed(
                cause="Failed to call backend API",
            )
            yield ErrorStreamPayload.from_error_response(error_response).serialize_json()
            return

        if agent_query.run_result is None:
            return

        usage = agent_query.run_result.usage
        yield EndStreamPayload.create(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        ).serialize_json()

    return _stream(), agent_query
