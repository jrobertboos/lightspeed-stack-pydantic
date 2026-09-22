## Providers

LLM providers are configured client-side and resolved through a registry onto
[pydantic-ai providers](https://ai.pydantic.dev/models/overview/).

```yaml
providers:
  - name: <name of provider>
    type: <type of provider>
    url: <base url>
    api_key: <api key>
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | yes | Unique registry key used at runtime to select this provider |
| `type` | yes | One of: `openai`, `azure`, `bedrock`, `vertexai`, `watsonx`, `vllm` |
| `url` | no | Base URL; mapped to `base_url`, `api_base`, or `azure_endpoint` as needed |
| `api_key` | no | API key; when omitted, pydantic-ai uses the provider's usual environment variables |

Supported `type` values and their pydantic-ai backends:

| type | Provider | Model kind |
| --- | --- | --- |
| `openai` | `OpenAIProvider` | Responses (`openai`) |
| `azure` | `AzureProvider` | Chat (`azure`) |
| `bedrock` | `BedrockProvider` | Converse (`bedrock`) |
| `vertexai` | `GoogleCloudProvider` | Google (`google-cloud`) |
| `watsonx` | `LiteLLMProvider` | Chat (`litellm`) |
| `vllm` | `OpenAIProvider` (compatible) | Chat (`openai-chat`) |

At runtime, `ProviderRegistry.get_model(name, model_name)` returns a pydantic-ai
`Model` bound to the registered provider (for a future agent loader).

Example:

```yaml
providers:
  - name: openai
    type: openai
    api_key: sk-...

  - name: my-vllm
    type: vllm
    url: http://localhost:8000/v1
    api_key: not-needed

  - name: watsonx
    type: watsonx
    api_key: ...
```

## MCP

MCP (Model Context Protocol) servers give the agent tools beyond what a
provider's model exposes natively. Only servers listed here are available to
agents; nothing from an external tool registry is discovered automatically.

Each entry becomes an [`MCP` capability](https://ai.pydantic.dev/capabilities/)
(`pydantic_ai.capabilities.MCP`), the recommended pydantic-ai entry point for
MCP, backed by a locally-run `MCPToolset` -- see
`lightspeed/core/agent/tools/mcp/factory.py`'s `MCPCapabilityFactory` and
`lightspeed/app/models/config.py`'s `MCPServerConfiguration`. Its tools show
up under a toolset labeled `mcp:<name>` (e.g. via `GET /tools`).

```yaml
mcp_servers:
  - name: <server name>
    url: <server url>
    timeout: <seconds>
    forward_headers:
      - <incoming request header name>
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | yes | Unique server name |
| `url` | yes | URL of the MCP server |
| `timeout` | no | Request timeout in seconds |
| `forward_headers` | no | Header names forwarded verbatim from the incoming client request (see below) |

Sending custom headers (e.g. an auth token) to an MCP server isn't
implemented yet.

### `forward_headers`

Header names listed here are copied from the incoming HTTP request and added
to every call to that MCP server, unmodified -- useful when infrastructure in
front of Lightspeed Stack (e.g. a gateway) injects headers like
`x-rh-identity` that the MCP server needs for user identification. Header
matching is case-insensitive and duplicates are rejected.

Not yet implemented -- this needs the incoming request's headers threaded
into the agent factory. A non-empty `forward_headers` raises
`NotImplementedError` when the agent is built.

Example:

```yaml
mcp_servers:
  - name: docs-search
    url: http://docs-mcp:8000

  - name: internal-api
    url: http://internal-api-mcp:8000
    timeout: 30
```

## Skills

Agent skills are portable `SKILL.md` packages following the
[Agent Skills](https://agentskills.io/specification) spec. Only paths listed
here are available to agents; nothing is discovered from a default
`.agents/skills` directory.

All configured paths become one [`Skills` capability](https://pydantic.dev/docs/ai/harness/skills/)
(`pydantic_ai_harness.skills.Skills`) -- see
`lightspeed/core/agent/tools/skills/factory.py`'s `SkillsCapabilityFactory`
and `lightspeed/app/models/config.py`'s `SkillsConfiguration`. The model first
sees each skill's name and description; calling `load_capability` injects
that skill's Markdown body.

Each path must be a **library**: a directory whose immediate children contain
`SKILL.md`. A path that points at a skill package itself is rejected.

```yaml
skills:
  paths:
    - <skill library path>
```

| Field | Required | Description |
| --- | --- | --- |
| `paths` | no | Skill libraries (directories of skill packages) |

Relative paths resolve from the process working directory. Discovery happens
when the capability is built (each `AgentFactory.create_agent` call). Missing
paths, invalid `SKILL.md` frontmatter, and duplicate skill names fail at
that point.

Harness `Skills` does not load bundled `references/` or `scripts/` files.

Example:

```yaml
skills:
  paths:
    - /var/skills
    - /opt/custom-skills
```

## Safety

Safety shields guard agent input (the user's prompt) and, for some shield
types, streamed model output. Each entry becomes one safety capability -- see
`lightspeed/core/agent/safety/factory.py`'s `SafetyCapabilityFactory` and
`lightspeed/app/models/config.py`'s `SafetyCapabilityConfiguration` -- built
via `AgentFactory.create_agent` alongside MCP and Skills capabilities.

```yaml
safety:
  - name: <shield name>
    type: <question_validity or redaction>
    config: <config required for type>
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | yes | Unique shield name |
| `type` | yes | One of: `question_validity`, `redaction` |
| `config` | no | Type-specific configuration (see below); omitted fields fall back to the shield's own defaults |

### `question_validity`

An LLM-classified on/off-topic guard for inbound prompts -- see
`QuestionValidityCapability`. Rejects prompts the classifier judges invalid,
off-topic, or an attempt to override instructions.

`config`:

| Field | Required | Description |
| --- | --- | --- |
| `model` | no | `<provider>:<model>` -- provider registry name and model id, resolved at build time. Omit to classify against the run's own model |
| `invalid_question_response` | no | Message returned to the caller in place of a rejected prompt |
| `classifier_instructions` | no | System prompt sent with the classification request |

```yaml
safety:
  - name: on-topic
    type: question_validity
    config:
      model: openai:gpt-4o-mini
      invalid_question_response: "I can only answer questions about our product."
```

### `redaction`

Regex-based PII redaction for prompts and model output -- see
`RedactionCapability`. Never blocks a request; it only rewrites matched
spans. Guards both prompt and output by default.

`config`:

| Field | Required | Description |
| --- | --- | --- |
| `patterns` | no | Named regex patterns to redact, keyed by a label. Defaults to a small set of common PII patterns (email, SSN, credit card, phone) |
| `replacement` | no | Text substituted in place of each match. Defaults to `[REDACTED]` |

```yaml
safety:
  - name: pii-redaction
    type: redaction
    config:
      replacement: "[hidden]"
      patterns:
        email: "[\\w.+-]+@[\\w-]+\\.[\\w.-]+"
```

Shields run in configuration order; a request that is `block`ed by an earlier
shield never reaches a later one.