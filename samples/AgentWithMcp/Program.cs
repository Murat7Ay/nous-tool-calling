// MCP: ListToolsAsync() → McpClientTool — all server tools are registered (no filtering).
// Env: OPENAI_* (same as ConsoleTest). MCP_ENDPOINT required to run.

using System.ClientModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using NousToolCalling.AgentFramework;
using OpenAI;

var endpoint = Environment.GetEnvironmentVariable("OPENAI_ENDPOINT")
    ?? throw new InvalidOperationException("Set OPENAI_ENDPOINT.");
var modelName = Environment.GetEnvironmentVariable("OPENAI_MODEL")
    ?? throw new InvalidOperationException("Set OPENAI_MODEL.");
var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "not-needed";

var mcpUrl = Environment.GetEnvironmentVariable("MCP_ENDPOINT");

if (string.IsNullOrWhiteSpace(mcpUrl))
{
    Console.WriteLine("Skip: set MCP_ENDPOINT to an MCP HTTP URL (streamable HTTP / SSE per SDK).");
    return;
}

IChatClient leafClient = new OpenAIClient(
        new ApiKeyCredential(apiKey),
        new OpenAIClientOptions { Endpoint = new Uri(endpoint) })
    .GetChatClient(modelName)
    .AsIChatClient();

IChatClient pipeline = NousChatClientPipeline.Create(leafClient);

await using var mcpHost = new McpToolHost();
IReadOnlyList<McpClientTool> mcpTools = await mcpHost.ListMcpToolsAsync(mcpUrl);
List<AITool> tools = mcpTools.Cast<AITool>().ToList();

if (tools.Count == 0)
{
    throw new InvalidOperationException($"MCP server returned no tools: {mcpUrl}");
}

AIAgent agent = pipeline.AsAIAgent(
    instructions: "You are a helpful assistant. Use MCP tools when they match the user's request.",
    name: "McpNousAgent",
    tools: tools);

Console.WriteLine(await agent.RunAsync(
    "What tools do you have? If there is a simple demo tool, invoke it once with minimal arguments."));
