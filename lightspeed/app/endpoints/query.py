"""Handler for REST API call to provide an answer to a query.

Minimal rewrite of the original Lightspeed ``/query`` endpoint. Inference goes
through :mod:`~lightspeed.core.agent.service`, which wraps
:meth:`~lightspeed.core.agent.factory.AgentFactory.create_agent` and
pydantic-ai ``Agent.run``. Flows that are not yet implemented are left as
comments.
"""

from __future__ import annotations

import logging
from typing import Any, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from lightspeed.app.models.requests.query import QueryRequest
from lightspeed.app.models.responses.error import (
    InternalServerErrorResponse,
    NotFoundResponse,
    UnprocessableEntityResponse,
)
from lightspeed.app.models.responses.success.query import QueryResponse
from lightspeed.core import service as core_service
from lightspeed.core.agent import service as agent_service
from lightspeed.core.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])

query_response: dict[int | str, dict[str, Any]] = {
    404: NotFoundResponse.openapi_response(
        examples=["conversation", "model", "provider"]
    ),
    422: UnprocessableEntityResponse.openapi_response(),
    500: InternalServerErrorResponse.openapi_response(
        examples=["configuration", "query"]
    ),
    # Not yet raised by this minimal rewrite:
    # 401: UnauthorizedResponse.openapi_response(...)
    # 403: ForbiddenResponse.openapi_response(...)
    # 413: PromptTooLongResponse.openapi_response(...)
    # 429: QuotaExceededResponse.openapi_response()
    # 503: ServiceUnavailableResponse.openapi_response(...)
}


@router.post(
    "/query",
    response_model=None,
    responses=query_response,
    summary="Query Endpoint Handler",
)
# @authorize(Action.QUERY)  # authorization not yet implemented
async def query_endpoint_handler(
    query_request: QueryRequest,
    # auth: Annotated[AuthTuple, Depends(get_auth_dependency())],  # auth not yet implemented
) -> Union[QueryResponse, StreamingResponse]:
    """Handle ``POST /query`` using a pydantic-ai agent.

    Implemented today:
        - Resolve ``Configuration`` from the process-wide singleton
        - ``AgentFactory.create_agent`` + ``agent.run`` for the user query
        - Return a minimal :class:`QueryResponse`, or an SSE
          :class:`~fastapi.responses.StreamingResponse` when
          ``query_request.stream`` is set
        - Structured error responses (``detail.response`` / ``detail.cause``)

    Not yet implemented (see commented steps below):
        - Auth / authorization / MCP OAuth
        - Quota checks and token accounting
        - Conversation load/store, compaction, topic summary
        - Shield moderation, RAG, tools/MCP
    """

    try:
        core_service.get_configuration()
    except RuntimeError as exc:
        error_response = InternalServerErrorResponse.configuration_not_loaded()
        raise HTTPException(**error_response.model_dump()) from exc

    try:
        agent = agent_service.create_agent(
            provider=query_request.provider,
            model=query_request.model,
            instructions=query_request.system_prompt,
        )
    except KeyError as exc:
        if query_request.provider not in ProviderRegistry():
            error_response = NotFoundResponse(
                resource="provider",
                resource_id=query_request.provider,
            )
        else:
            error_response = NotFoundResponse(
                resource="model",
                resource_id=query_request.model,
            )
        raise HTTPException(**error_response.model_dump()) from exc
    except Exception as exc:
        logger.exception("Failed to create agent")
        error_response = InternalServerErrorResponse.query_failed(
            cause="Failed to create agent",
        )
        raise HTTPException(**error_response.model_dump()) from exc

    if query_request.stream:
        try:
            stream, _agent_query = await agent_service.stream_query(
                agent,
                query_request.query,
                conversation_id=query_request.conversation_id,
            )
        except Exception as exc:
            logger.exception("Failed to stream query")
            error_response = InternalServerErrorResponse.query_failed(
                cause="Failed to stream query",
            )
            raise HTTPException(**error_response.model_dump()) from exc
        return StreamingResponse(stream, media_type="text/event-stream")

    try:
        agent_query = await agent_service.query(
            agent,
            query_request.query,
            conversation_id=query_request.conversation_id,
        )
    except ValueError as exc:
        error_response = UnprocessableEntityResponse(
            response="Invalid attribute value",
            cause=str(exc),
        )
        raise HTTPException(**error_response.model_dump()) from exc
    except Exception as exc:
        logger.exception("Failed to query")
        error_response = InternalServerErrorResponse.query_failed(
            cause="Failed to query",
        )
        raise HTTPException(**error_response.model_dump()) from exc

    # topic_summary = await maybe_get_topic_summary(...)
    # consume_query_tokens(...)
    # available_quotas = get_available_quotas(...)
    # store_query_results(...)

    return QueryResponse.from_agent_query(agent_query)