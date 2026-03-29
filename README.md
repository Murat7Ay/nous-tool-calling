# NousToolCalling

Prompt-based (Nous-style) tool calling for .NET `IChatClient` pipelines.

Translates between `FunctionCallContent`/`FunctionResultContent` (the standard `Microsoft.Extensions.AI` protocol) and the **XML-wrapped tool call format** used by models like Qwen, DeepSeek, and other open-source LLMs that don't support native OpenAI `tools`/`tool_calls` API.

## How It Works

Many open-source models use a prompt-driven tool calling convention (originating from the NousResearch fine-tunes):

1. **Tool schemas** are embedded in the system prompt inside `<tools>...</tools>` XML.
2. **Tool invocations** are returned by the model as `<tool_call>{"name":"...","arguments":{...}}</tool_call>` XML blocks in plain assistant text.
3. **Tool results** are sent back as `<tool_response>...</tool_response>` in user messages.

`NousToolCallingChatClient` is a `DelegatingChatClient` decorator that performs this translation automatically, so the rest of your pipeline (including `FunctionInvokingChatClient` for automatic tool execution) works unchanged.

## Installation

This project targets .NET 9+ and depends only on [`Microsoft.Extensions.AI`](https://www.nuget.org/packages/Microsoft.Extensions.AI/) (10.4.0).

```bash
dotnet build
```

## Usage

```csharp
using NousToolCalling;
using Microsoft.Extensions.AI;

IChatClient pipeline = yourLeafClient
    .AsBuilder()
    .UseFunctionInvocation()       // MUST be first (outermost layer)
    .UseNousToolCalling(new NousToolCallingOptions
    {
        StrictThinkMode = true,      // defer parsing while <think> is open
        PreserveThinkBlocks = true,  // store <think> content in AdditionalProperties
        ToolCallIdPrefix = "nous",   // prefix for generated call IDs
    })
    .Build();

var tools = new List<AITool>
{
    AIFunctionFactory.Create(
        (string city) => $"Sunny, 22C in {city}",
        name: "get_weather",
        description: "Gets weather for a city"),
};

var response = await pipeline.GetResponseAsync(
    [new ChatMessage(ChatRole.User, "What's the weather in Istanbul?")],
    new ChatOptions { Tools = tools });
```

## Pipeline Order (Critical)

> **`UseFunctionInvocation()` MUST come before `UseNousToolCalling()` in the builder chain.**

`ChatClientBuilder` applies `Use()` calls in reverse order — the first call becomes the **outermost** decorator. `FunctionInvokingChatClient` must be outermost so it can see the `FunctionCallContent` instances that `NousToolCallingChatClient` produces.

```
Request flow:  User → FunctionInvokingChatClient → NousToolCallingChatClient → Leaf (OpenAI/Qwen)
Response flow: Leaf → NousToolCallingChatClient (parses XML → FunctionCallContent) → FunctionInvokingChatClient (invokes tools)
```

If the order is reversed, `FunctionInvokingChatClient` will see raw XML text instead of structured `FunctionCallContent` and will not invoke any tools.

## Configuration

| Option | Default | Description |
|---|---|---|
| `StrictThinkMode` | `true` | Defer tool-call extraction while a `<think>` block is open (prevents false positives during model reasoning) |
| `PreserveThinkBlocks` | `false` | Store `<think>` content in `message.AdditionalProperties["nous_think"]` |
| `ToolCallIdPrefix` | `"nous"` | Prefix for generated synthetic call IDs (`{prefix}_{ordinal}`) |

## Features

- Converts `ChatOptions.Tools` to Nous `<tools>` system prompt section
- Parses `<tool_call>` XML blocks from assistant responses into `FunctionCallContent`
- Sets `FinishReason = ChatFinishReason.ToolCalls` so `FunctionInvokingChatClient` processes them
- Converts `FunctionCallContent` back to `<tool_call>` XML on outbound requests (multi-turn)
- Converts `FunctionResultContent` to `<tool_response>` XML in user messages
- Supports `<think>` block handling (strict and lenient modes)
- Works with both `GetResponseAsync` and `GetStreamingResponseAsync`

## Project Structure

```
nous-tool-calling/
├── src/NousToolCalling/              # Core library
│   ├── NousToolCallingChatClient.cs  # Main DelegatingChatClient decorator
│   ├── NousToolCallParser.cs         # Stateless XML tool-call parser
│   ├── NousToolPromptBuilder.cs      # Builds <tools> prompt section
│   ├── NousToolCallingOptions.cs     # Configuration
│   └── NousToolCallingChatClientExtensions.cs  # UseNousToolCalling() extension
├── tests/NousToolCalling.UnitTests/  # 27 unit tests
├── samples/ConsoleTest/              # Interactive console demo
├── NousToolCalling.slnx
└── README.md
```

## Running Tests

```bash
dotnet test
```

## Running the Console Sample

The console sample connects to an OpenAI-compatible endpoint with dummy tools (weather, search, calculator):

```bash
# Set environment variables
export OPENAI_API_KEY="your-key"
export OPENAI_ENDPOINT="https://api.openai.com/v1"  # or your local endpoint
export OPENAI_MODEL="gpt-4o-mini"                    # or your model name

# Run
dotnet run --project samples/ConsoleTest
```

## License

Copyright (c) Microsoft. All rights reserved.
