// CopilotKit hosted MCP (https://mcp.copilotkit.ai/mcp) + Nous/Qwen-ready ChatClientAgent.
//
// Usage:
//   dotnet run -- --list-mcp-tools     → only hits MCP; needs network. OPENAI_* not required.
//   dotnet run                         → full agent; set OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY.
//
// MCP URL: set MCP_ENDPOINT or it defaults to CopilotKit's public endpoint below.

using System.ClientModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using NousToolCalling.AgentFramework;
using OpenAI;

const string DefaultCopilotKitMcp = "https://mcp.copilotkit.ai/mcp";

bool listOnly = args.Contains("--list-mcp-tools", StringComparer.OrdinalIgnoreCase);

var mcpUrl = Environment.GetEnvironmentVariable("MCP_ENDPOINT");
if (string.IsNullOrWhiteSpace(mcpUrl))
{
    mcpUrl = DefaultCopilotKitMcp;
    Console.WriteLine($"Using default MCP_ENDPOINT: {mcpUrl}");
}

await using (var mcpHost = new McpToolHost())
{
    IReadOnlyList<McpClientTool> mcpTools;
    try
    {
        mcpTools = await mcpHost.ListMcpToolsAsync(mcpUrl);
    }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"MCP ListTools failed: {ex.Message}");
        Environment.ExitCode = 1;
        return;
    }

    Console.WriteLine($"MCP tools ({mcpTools.Count}):");
    foreach (var t in mcpTools)
    {
        Console.WriteLine($"  - {t.Name}: {t.Description}");
    }

    if (listOnly)
    {
        return;
    }

    var endpoint = Environment.GetEnvironmentVariable("OPENAI_ENDPOINT")
        ?? throw new InvalidOperationException("Set OPENAI_ENDPOINT for full agent run (or use --list-mcp-tools).");
    var modelName = Environment.GetEnvironmentVariable("OPENAI_MODEL")
        ?? throw new InvalidOperationException("Set OPENAI_MODEL.");
    var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "not-needed";

    List<AITool> tools = mcpTools.Cast<AITool>().ToList();

    IChatClient leafClient = new OpenAIClient(
            new ApiKeyCredential(apiKey),
            new OpenAIClientOptions { Endpoint = new Uri(endpoint) })
        .GetChatClient(modelName)
        .AsIChatClient();

    IChatClient pipeline = NousChatClientPipeline.Create(leafClient);

    AIAgent agent = pipeline.AsAIAgent(
        instructions:
        "You are a helpful assistant. Use MCP tools when they help answer the user. Be concise.",
        name: "CopilotKitMcpAgent",
        tools: tools);

    Console.WriteLine();
    Console.WriteLine("--- Agent run ---");
    var prompt = Environment.GetEnvironmentVariable("AGENT_PROMPT")
        ?? "List what MCP tools you have access to. If one looks safe to demo, call it with minimal arguments.";
    Console.WriteLine(await agent.RunAsync(prompt));
}
