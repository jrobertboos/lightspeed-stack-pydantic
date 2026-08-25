"""Configuration models for Lightspeed Stack."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, PositiveInt, SecretStr

# Supported inference provider types.
ProviderType = Literal[
    "openai",
    "azure",
    "bedrock",
    "vertexai",
    "watsonx",
    "vllm",
]


class ConfigurationBase(BaseModel):
    """Base class for configuration models that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class ProviderConfiguration(ConfigurationBase):
    """Client-side LLM provider entry.

    ``type`` selects a supported Lightspeed provider. The registry maps each
    type onto a pydantic-ai :class:`~pydantic_ai.providers.Provider`.

    YAML shape::

        providers:
          - name: my-openai
            type: openai
            url: https://api.openai.com/v1
            api_key: sk-...
    """

    name: str = Field(
        ...,
        title="Provider name",
        description="Unique identifier used to look up this provider in the registry.",
        min_length=1,
    )

    type: ProviderType = Field(
        ...,
        title="Provider type",
        description=(
            "Supported provider type: openai, azure, bedrock, vertexai, watsonx, or vllm."
        ),
    )

    url: Optional[AnyHttpUrl] = Field(
        None,
        title="Base URL",
        description=(
            "Optional base URL for the provider API. Mapped to the underlying "
            "pydantic-ai provider as base_url, api_base, or azure_endpoint."
        ),
    )

    api_key: Optional[SecretStr] = Field(
        None,
        title="API key",
        description=(
            "Optional API key. When omitted, the underlying pydantic-ai provider "
            "falls back to its standard environment variables."
        ),
    )


class ServiceConfiguration(ConfigurationBase):
    """Service configuration.

    Controls how the REST API service binds and how many Uvicorn worker
    processes handle requests concurrently. TLS and CORS (present in the
    original Lightspeed Stack) are not yet implemented.
    """

    host: str = Field(
        "localhost",
        title="Host",
        description="Service hostname.",
    )

    port: PositiveInt = Field(
        8080,
        title="Port",
        description="Service port.",
    )

    workers: PositiveInt = Field(
        1,
        title="Workers",
        description="Number of Uvicorn worker processes.",
    )

    auth_enabled: bool = Field(
        False,
        title="Auth enabled",
        description="Enable authentication. Not yet implemented.",
    )

    color_log: bool = Field(
        True,
        title="Color log",
        description="Enable colorized console logging. Not yet implemented.",
    )

    access_log: bool = Field(
        True,
        title="Access log",
        description="Enable Uvicorn access logging.",
    )


class Configuration(ConfigurationBase):
    """Root Lightspeed Stack configuration."""

    name: str = Field(
        "Lightspeed Stack",
        title="Service name",
        description="Name of the service.",
    )

    service: ServiceConfiguration = Field(
        default_factory=ServiceConfiguration,
        title="Service",
        description="REST API service configuration (host, port, workers, ...).",
    )

    providers: list[ProviderConfiguration] = Field(
        default_factory=list,
        title="Providers",
        description="Configured LLM providers available to the runtime.",
    )

    # Sections present in lightspeed-stack.yaml that are not yet implemented in
    # this rewrite. Declared here (instead of relying on `extra="forbid"`
    # rejecting them) so a full configuration file loads without error; each
    # becomes a typed model as its feature lands.
    authentication: Optional[Any] = Field(None, title="Authentication")
    authorization: Optional[Any] = Field(None, title="Authorization")
    knowledge: Optional[Any] = Field(None, title="Knowledge")
    safety: Optional[Any] = Field(None, title="Safety")
    mcp_servers: Optional[Any] = Field(None, title="MCP servers")
    skills: Optional[Any] = Field(None, title="Skills")
