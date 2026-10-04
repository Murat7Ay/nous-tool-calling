# NousToolCalling.AgentFramework

Microsoft Agent Framework helpers on top of [NousToolCalling](https://www.nuget.org/packages/NousToolCalling), for on-prem / OpenAI-compatible models that emit `<tool_call>` XML instead of native `tool_calls`.

- **`NousChatClientPipeline.Create(leafClient)`** builds the correct `FunctionInvocation → NousToolCalling → leaf` pipeline in one call.
- **`McpToolHost`** connects to MCP servers over HTTP and exposes all their tools as `AITool`s (`McpClientTool`), keeping the server's JSON schemas.

```csharp
using Microsoft.Agents.AI;
using NousToolCalling.AgentFramework;

IChatClient chat = NousChatClientPipeline.Create(leafClient); // any OpenAI-compatible IChatClient

await using var mcp = new McpToolHost();
var tools = await mcp.ListAIToolsAsync("https://mcp.example.com/mcp");

AIAgent agent = chat.AsAIAgent(instructions: "You are a helpful assistant.", tools: [.. tools]);
Console.WriteLine(await agent.RunAsync("Use the tools to answer: ..."));
```

Samples (RAG, multi-agent workflows, sub-agents as tools, OpenTelemetry, sessions): https://github.com/Murat7Ay/nous-tool-calling
