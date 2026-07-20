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
