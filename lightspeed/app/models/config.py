"""Configuration models for Lightspeed Stack."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    SecretStr,
    field_validator,
)

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


class MCPServerConfiguration(ConfigurationBase):
    """MCP (Model Context Protocol) server entry.

    Each entry becomes an ``MCP`` capability (see
    :class:`~lightspeed.core.agent.tools.mcp.factory.MCPCapabilityFactory`)
    that gives the agent tools beyond what a provider's model exposes
    natively. Only servers listed here are available to agents.

    YAML shape::

        mcp_servers:
          - name: docs-search
            url: http://docs-mcp:8000
            timeout: 30
            forward_headers:
              - x-rh-identity
    """

    name: str = Field(
        ...,
        title="Server name",
        description="Unique identifier for this MCP server.",
        min_length=1,
    )

    url: AnyHttpUrl = Field(
        ...,
        title="Server URL",
        description="URL of the MCP server.",
    )

    timeout: Optional[PositiveInt] = Field(
        None,
        title="Timeout",
        description="Request timeout in seconds.",
    )

    forward_headers: list[str] = Field(
        default_factory=list,
        title="Forwarded headers",
        description=(
            "Header names forwarded verbatim from the incoming client "
            "request to this MCP server (not yet implemented -- needs "
            "per-request context that isn't wired up yet)."
        ),
    )

    @field_validator("forward_headers")
    @classmethod
    def validate_forward_headers(cls, value: list[str]) -> list[str]:
        """Reject blank header names and case-insensitive duplicates."""
        seen: set[str] = set()
        for header in value:
            if not header.strip():
                raise ValueError("forward_headers entries must not be blank")
            lowered = header.lower()
            if lowered in seen:
                raise ValueError(f"Duplicate forward_headers entry: {header!r}")
            seen.add(lowered)
        return value


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

    mcp_servers: list[MCPServerConfiguration] = Field(
        default_factory=list,
        title="MCP servers",
        description="Configured MCP servers available to agents as tools.",
    )

    # Sections present in lightspeed-stack.yaml that are not yet implemented in
    # this rewrite. Declared here (instead of relying on `extra="forbid"`
    # rejecting them) so a full configuration file loads without error; each
    # becomes a typed model as its feature lands.
    authentication: Optional[Any] = Field(None, title="Authentication")
    authorization: Optional[Any] = Field(None, title="Authorization")
    knowledge: Optional[Any] = Field(None, title="Knowledge")
    safety: Optional[Any] = Field(None, title="Safety")
    skills: Optional[Any] = Field(None, title="Skills")
