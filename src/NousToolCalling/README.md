# NousToolCalling

Make on-prem / OpenAI-compatible LLMs that emit tool calls as **XML text** work with `Microsoft.Extensions.AI` and Microsoft Agent Framework tools.

Many self-hosted models (Qwen, Hermes, DeepSeek, …) served via vLLM, llama.cpp, Ollama or a corporate gateway don't fill the OpenAI `tool_calls` field. They reply with plain text like:

```text
<tool_call>{"name": "get_weather", "arguments": {"city": "Istanbul"}}</tool_call>
```

`FunctionInvokingChatClient` never sees that, so your tools never run. `NousToolCallingChatClient` is a `DelegatingChatClient` that:

- injects your `ChatOptions.Tools` into the system prompt as `<tools>` JSON schemas,
- parses `<tool_call>` blocks (streaming and non-streaming) into `FunctionCallContent`,
- sends tool results back as `<tool_response>` messages,
- handles `<think>` reasoning blocks safely.

## Usage

```csharp
using Microsoft.Extensions.AI;

IChatClient client = new OpenAI.Chat.ChatClient(
        model: "qwen3-32b",
        credential: new System.ClientModel.ApiKeyCredential("not-needed"),
        options: new OpenAI.OpenAIClientOptions { Endpoint = new Uri("http://localhost:8080/v1") })
    .AsIChatClient()
    .AsBuilder()
    .UseFunctionInvocation()   // must come first (outermost)
    .UseNousToolCalling()
    .Build();

var response = await client.GetResponseAsync(
    "What's the weather in Istanbul?",
    new ChatOptions { Tools = [AIFunctionFactory.Create((string city) => $"Sunny, 22C in {city}", "get_weather")] });
```

`client` can also be wrapped in a `ChatClientAgent` (`client.AsAIAgent(...)`), so agents, RAG and MCP tools work unchanged.

## Options

| Option | Default | Description |
|---|---|---|
| `StrictThinkMode` | `true` | Don't parse tool calls while a `<think>` block is still open |
| `PreserveThinkBlocks` | `false` | Store `<think>` content in `AdditionalProperties["nous_think"]` |
| `ToolCallIdPrefix` | `"nous"` | Prefix for generated call IDs |

Docs, samples and source: https://github.com/Murat7Ay/nous-tool-calling
