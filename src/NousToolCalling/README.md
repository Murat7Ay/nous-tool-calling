# NousToolCalling

Make on-prem / OpenAI-compatible LLMs that emit tool calls as **XML text** work with `Microsoft.Extensions.AI` and Microsoft Agent Framework tools.

Many self-hosted models (Qwen, Hermes, DeepSeek, GLM, …) served via vLLM, llama.cpp, Ollama or a corporate gateway don't fill the OpenAI `tool_calls` field. They reply with plain text like:

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
| `ToolCallIdPrefix` | `"nous"` | Prefix for generated call IDs (unique per response) |
| `ToolCallFormat` | `Hermes` | `Hermes` (JSON in `<tool_call>`), `Glm` (`<arg_key>`/`<arg_value>`), or `Auto` |

## GLM models (GLM-4.5 … GLM-5.3-Flash) on vLLM

Written against vLLM 0.30.0 (`glm45` / `glm47` tool and reasoning parsers) and the zai-org GLM-4.5, 4.6, 4.7, 4.7-Flash, 5.x and 5.3-Flash chat templates.

**1. Normalize the HTTP traffic.** `GlmNormalizationHandler` is an `HttpMessageHandler` that fixes the raw `/chat/completions` JSON before any SDK reads it, so it works with the OpenAI .NET SDK and `Microsoft.Extensions.AI.OpenAI`:

```csharp
using System.ClientModel.Primitives;
using NousToolCalling;

var http = new HttpClient(new GlmNormalizationHandler(new SocketsHttpHandler()));  // optional: GlmNormalizationOptions, ILogger

IChatClient leaf = new OpenAI.OpenAIClient(
        new System.ClientModel.ApiKeyCredential("not-needed"),
        new OpenAI.OpenAIClientOptions { Endpoint = new Uri("http://localhost:8000/v1"), Transport = new HttpClientPipelineTransport(http) })
    .GetChatClient("zai-org/GLM-5.3-Flash")
    .AsIChatClient();
```

| Option | Default | What it does |
|---|---|---|
| `StripReasoningFromHistory` | `true` | Removes `reasoning` / `reasoning_content` from the messages sent back to the server |
| `EnableThinking` | `null` | `null` removes `chat_template_kwargs.enable_thinking` from requests; set it to send it. GLM-5.3 always thinks, and `false` makes the reasoning leak into the content |
| `StripThinkTags` | `true` | Moves `<think>…</think>` (or text before a lone `</think>`) out of non-streaming content |
| `ReasoningAsContentFallback` | `true` | When a choice ends with empty content and no tool calls, the reasoning becomes the answer (streaming and non-streaming). Reasoning deltas are joined with nothing in between |
| `MirrorReasoningContent` | `true` | Copies `reasoning` to `reasoning_content`, which `Microsoft.Extensions.AI.OpenAI` surfaces as `TextReasoningContent` |
| `DecodeUnicodeEscapes` | `true` | Decodes literal `\u00fc` / `\ud83d\ude00` text in content and reasoning; code points below 0x20 stay as written; escapes split across stream chunks are held until complete |
| `DropEmptyChoicesChunks` | `true` | Drops the final usage-only chunk whose `choices` is empty |
| `HandleUpstreamErrors` | `true` | Closes a stream that breaks mid-answer cleanly, turns connection failures into HTTP 502 with an OpenAI-style error body, and wraps bare `{"object":"error",…}` bodies in `{"error": …}` |

A warning is logged when a choice ends with `finish_reason: "length"` and no content: `max_tokens` was too low and the model was cut off while thinking. Connection settings such as timeouts belong on the inner handler you pass in.

**2. Tool calls.** Which setup you need depends on how vLLM is started:

- **vLLM's GLM tool parser on** (`--enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45`): the server returns native `tool_calls`. Use the handler and `UseFunctionInvocation()` only; no `UseNousToolCalling()` needed.
- **Tool parser off**: the model writes calls as text, `<tool_call>get_weather<arg_key>city</arg_key><arg_value>Istanbul</arg_value></tool_call>`. Add `UseNousToolCalling` with the GLM format:

```csharp
IChatClient client = leaf.AsBuilder()
    .UseFunctionInvocation()
    .UseNousToolCalling(new NousToolCallingOptions { ToolCallFormat = NousToolCallFormat.Glm })
    .Build();
```

The GLM format writes the tools prompt and the tool-call history the way the GLM chat templates do, accepts GLM-4.5 (newline-separated) and GLM-4.7+ (single-line) calls, calls without arguments, several calls per reply, and tags split across stream chunks. Argument values are text; they are converted to numbers, booleans, objects and arrays using the tool's JSON schema. Output that carries only a closing `</think>` (GLM-4.7+ prompts already open the block) is treated as reasoning and never blocks streaming.

`NousToolCallFormat.Auto` accepts both Hermes and GLM calls in the output and picks the GLM prompt when the model id contains `glm`.

Docs, samples and source: https://github.com/Murat7Ay/nous-tool-calling
