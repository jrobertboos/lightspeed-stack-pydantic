"""Handler for REST API call to list available models.

Minimal rewrite of the original Lightspeed ``/models`` endpoint. There's no
Llama Stack model catalog to query; models come from the process-wide
:class:`~lightspeed.core.providers.registry.ProviderRegistry`, which is
populated by :meth:`ProviderRegistry.load` during app startup (see
:mod:`lightspeed.app.main`). Auth/authorization and the original
``model_type`` query filter (llm/embedding) are not yet implemented.
"""

from __future__ import annotations

from fastapi import APIRouter

from lightspeed.app.models.responses.success.models import ModelInfo, ModelsResponse
from lightspeed.core.providers.registry import ProviderRegistry

router = APIRouter(tags=["models"])


@router.get("/models", summary="Models Endpoint Handler")
async def models_endpoint_handler() -> ModelsResponse:
    """Return the models available across all configured providers.

    Reads directly from :class:`ProviderRegistry`, which eagerly lists each
    provider's models at startup, so this endpoint makes no client calls of
    its own.
    """
    registry = ProviderRegistry()
    models = [
        ModelInfo(
            identifier=f"{provider_name}/{model.model_name}",
            provider_name=provider_name,
            model_name=model.model_name,
        )
        for provider_name, provider_models in registry.models.items()
        for model in provider_models
    ]
    return ModelsResponse(models=models)
