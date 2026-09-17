"""Configuration models for Lightspeed Stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    SecretStr,
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

# Supported knowledge (RAG) source backend types. Only the shape is
# implemented so far -- none of these has a concrete
# :class:`~lightspeed.core.agent.knowledge.store.VectorStore` builder yet, so
# a configured source needs a store registered by hand via
# :class:`~lightspeed.core.agent.knowledge.registry.KnowledgeStoreRegistry`
# (see :mod:`lightspeed.core.agent.knowledge.factory`).
KnowledgeSourceType = Literal["faiss", "pgvector", "okp"]


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


class SkillsConfiguration(ConfigurationBase):
    """Agent skills configuration.

    Each path is a skill library: a directory whose immediate children
    contain ``SKILL.md``. All paths become a single
    :class:`~pydantic_ai_harness.skills.Skills` capability (see
    :class:`~lightspeed.core.agent.tools.skills.factory.SkillsCapabilityFactory`).

    YAML shape::

        skills:
          paths:
            - /var/skills
            - /opt/custom-skills
    """

    paths: list[Path] = Field(
        default_factory=list,
        title="Skill paths",
        description="Paths to skill libraries (directories of skill packages).",
    )


class KnowledgeSourceConfiguration(ConfigurationBase):
    """One knowledge (RAG) source entry.

    Each entry becomes one or two :class:`~lightspeed.core.agent.knowledge.capability.Knowledge`
    capabilities (see
    :class:`~lightspeed.core.agent.knowledge.factory.KnowledgeCapabilityFactory`),
    depending on whether ``name`` is listed under
    :attr:`KnowledgeConfiguration.strategy`'s ``inline`` and/or ``tool`` lists.

    ``type`` and ``config`` describe the backend a future
    :class:`~lightspeed.core.agent.knowledge.store.VectorStore` builder would
    connect to; there is no such builder yet, so a source is only usable once
    something calls
    :meth:`~lightspeed.core.agent.knowledge.registry.KnowledgeStoreRegistry.register`
    for its ``name``.

    YAML shape::

        knowledge:
          sources:
            - name: product-docs
              type: pgvector
              config:
                dsn: postgresql://...
              embedding_model: openai:text-embedding-3-small
              top_k: 5
    """

    name: str = Field(
        ...,
        title="Source name",
        description="Unique identifier for this knowledge source.",
        min_length=1,
    )

    type: KnowledgeSourceType = Field(
        ...,
        title="Source type",
        description=(
            "Backend type: faiss, pgvector, or okp. None has a built-in "
            "VectorStore builder yet -- see KnowledgeSourceConfiguration."
        ),
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
        title="Backend configuration",
        description=(
            "Backend-specific connection settings (e.g. a pgvector DSN). "
            "Not yet consumed by anything -- reserved for when `type` has a "
            "VectorStore builder."
        ),
    )

    embedding_model: str = Field(
        ...,
        title="Embedding model",
        description=(
            "'provider:model' embedding model identifier resolved via "
            "pydantic_ai.embeddings.Embedder, e.g. "
            "'openai:text-embedding-3-small'."
        ),
        min_length=1,
    )

    top_k: PositiveInt = Field(
        5,
        title="Top K",
        description="Number of matches returned per knowledge search.",
    )


class KnowledgeStrategyConfiguration(ConfigurationBase):
    """Assigns each knowledge source to one or both exposure modes.

    A source named in neither list still defaults to ``'tool'`` mode (see
    :class:`~lightspeed.core.agent.knowledge.factory.KnowledgeCapabilityFactory`);
    listing it under both makes it available both ways at once.
    """

    inline: list[str] = Field(
        default_factory=list,
        title="Inline sources",
        description=(
            "Source names automatically searched and injected into context "
            "on every model request."
        ),
    )

    tool: list[str] = Field(
        default_factory=list,
        title="Tool sources",
        description="Source names exposed as a model-callable search tool.",
    )


class RerankerConfiguration(ConfigurationBase):
    """Reranker configuration. Not yet implemented.

    Declared so a configuration file that sets it loads and validates, but
    :class:`~lightspeed.core.agent.knowledge.factory.KnowledgeCapabilityFactory`
    raises ``NotImplementedError`` if this is set.
    """

    model: str = Field(
        ...,
        title="Reranker model",
        description="Reranker model identifier. Not yet implemented.",
    )


class KnowledgeConfiguration(ConfigurationBase):
    """Knowledge (RAG) configuration.

    YAML shape::

        knowledge:
          sources:
            - name: product-docs
              type: pgvector
              config: {}
              embedding_model: openai:text-embedding-3-small
          strategy:
            tool:
              - product-docs
    """

    sources: list[KnowledgeSourceConfiguration] = Field(
        default_factory=list,
        title="Knowledge sources",
        description="Configured knowledge sources available to agents.",
    )

    strategy: Optional[KnowledgeStrategyConfiguration] = Field(
        None,
        title="Strategy",
        description="Assigns sources to inline and/or tool exposure modes.",
    )

    reranker: Optional[RerankerConfiguration] = Field(
        None,
        title="Reranker",
        description="Reranker configuration. Not yet implemented.",
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

    mcp_servers: list[MCPServerConfiguration] = Field(
        default_factory=list,
        title="MCP servers",
        description="Configured MCP servers available to agents as tools.",
    )

    skills: Optional[SkillsConfiguration] = Field(
        None,
        title="Agent skills",
        description=(
            "Agent skills configuration. Specifies paths to skill libraries."
        ),
    )

    knowledge: Optional[KnowledgeConfiguration] = Field(
        None,
        title="Knowledge",
        description=(
            "Knowledge (RAG) configuration. Specifies knowledge sources and "
            "how they're exposed to agents."
        ),
    )

    # Sections present in lightspeed-stack.yaml that are not yet implemented in
    # this rewrite. Declared here (instead of relying on `extra="forbid"`
    # rejecting them) so a full configuration file loads without error; each
    # becomes a typed model as its feature lands.
    authentication: Optional[Any] = Field(None, title="Authentication")
    authorization: Optional[Any] = Field(None, title="Authorization")
    safety: Optional[Any] = Field(None, title="Safety")
