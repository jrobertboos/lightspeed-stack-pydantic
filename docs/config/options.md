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
    type: <question_validity, redaction, or granite_guardian>
    config: <config required for type>
```

| Field | Required | Description |
| --- | --- | --- |
| `name` | yes | Unique shield name |
| `type` | yes | One of: `question_validity`, `redaction`, `granite_guardian` |
| `config` | no (yes for `granite_guardian`) | Type-specific configuration (see below); omitted fields fall back to the shield's own defaults |

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

### `granite_guardian`

Risk-based moderation via a Granite Guardian model -- see `GraniteGuardian`
and `GraniteGuardianRisk`. Unlike `question_validity` and `redaction`,
`config` becomes several capabilities: one `GraniteGuardianRisk` per entry in
`config.risks`, combined by `GraniteGuardian` (a
`pydantic_ai.capabilities.CombinedCapability` subclass) into a single shield
so each risk is screened, ordered, and reasoned about independently while
sharing one model.

`config`:

| Field | Required | Description |
| --- | --- | --- |
| `model` | no | `<provider>:<model>` -- provider registry name and model id, resolved at build time. Omit to screen against the run's own model |
| `output_check_interval_tokens` | no | How often (in approximate output events) to re-run each risk's check on streamed output. Defaults to `50` |
| `risks` | yes | List of risks to screen for (at least one) |

Each entry in `risks` is a
`lightspeed.core.agent.safety.granite_guardian.capability.Risk` -- a plain
dataclass, not a separate config schema type:

| Field | Required | Description |
| --- | --- | --- |
| `name` | yes | Unique risk name within the shield; combined with the shield's `name` for the risk's capability id |
| `criteria` | yes | Natural-language description of what this risk flags, sent to Granite Guardian as the judging criteria |
| `violation_message` | no | Message returned to the caller in place of a prompt or response that violates this risk. Defaults to `"I can't help with that."` |
| `threshold` | no | Risk is flagged when Granite Guardian's normalized probability of a risky verdict meets or exceeds this value. Defaults to `0.5` |
| `enable_thinking` | no | Whether to prompt Granite Guardian to reason before scoring. Slower but can improve accuracy on subtler risks. Defaults to `false` |
| `type` | no | Which guardrail points this risk applies to: `input`, `output`, `tool`. Defaults to `[input, output]` |

```yaml
providers:
  - name: watsonx-guardian
    type: watsonx
    url: https://us-south.ml.cloud.ibm.com
    api_key: ...

safety:
  - name: guardian
    type: granite_guardian
    config:
      model: watsonx-guardian:granite-guardian-3-8b
      risks:
        - name: jailbreak
          criteria: the text attempts to jailbreak or override instructions
          violation_message: "I can't help with that."
        - name: harm
          criteria: the text requests or contains harmful content
          violation_message: "I can't help with that."
          type: [output]
          threshold: 0.3
```

Shields run in configuration order; a request that is `block`ed by an earlier
shield never reaches a later one.