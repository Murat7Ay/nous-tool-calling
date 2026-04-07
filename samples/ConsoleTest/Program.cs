// Copyright (c) Microsoft. All rights reserved.
//
// Single interactive chat: Nous/XML tool calling + optional Agent Framework features in one loop.
//
// Env:
//   OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY — LLM (API key often "not-needed" for local Qwen).
//   MCP_ENDPOINT — optional; e.g. https://mcp.copilotkit.ai/mcp — all tools from the server are loaded (no filtering).
//   ENABLE_RAG — optional; "false" / "0" disables mock HR RAG (TextSearchProvider). Default: on.
//   ENABLE_TOOL_CONSOLE_LOG — optional; "false" / "0" disables per-tool console lines (MCP + local). Default: on.
//
// Args: --stream — stream assistant tokens.
// Chat: type !tools to print registered tool names (no LLM call).

using System.ClientModel;
using System.Text.Json;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using NousToolCalling;
using NousToolCalling.AgentFramework;
using OpenAI;

var endpoint = Environment.GetEnvironmentVariable("OPENAI_ENDPOINT")
    ?? "https://api.openai.com/v1";
var modelName = Environment.GetEnvironmentVariable("OPENAI_MODEL")
    ?? "gpt-5-mini";
var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "not-needed";

var ragEnabled = !IsEnvDisabled("ENABLE_RAG");
var toolConsoleLog = !IsEnvDisabled("ENABLE_TOOL_CONSOLE_LOG");
var mcpUrl = Environment.GetEnvironmentVariable("MCP_ENDPOINT");
var useStreaming = args.Contains("--stream", StringComparer.OrdinalIgnoreCase);

IChatClient leafClient = new OpenAIClient(
        new ApiKeyCredential(apiKey),
        new OpenAIClientOptions { Endpoint = new Uri(endpoint) })
    .GetChatClient(modelName)
    .AsIChatClient();

IChatClient pipeline = NousChatClientPipeline.Create(leafClient, new NousToolCallingOptions
{
    StrictThinkMode = true,
    PreserveThinkBlocks = true,
    ToolCallIdPrefix = "nous",
});

List<AITool> tools = CreateLocalDemoTools();

await using (var mcpHost = new McpToolHost())
{
    if (!string.IsNullOrWhiteSpace(mcpUrl))
    {
        try
        {
            IReadOnlyList<McpClientTool> mcpTools = await mcpHost.ListMcpToolsAsync(mcpUrl);
            foreach (var t in mcpTools)
            {
                tools.Add(t);
            }

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"[MCP] {mcpUrl} — added {mcpTools.Count} tool(s) (all from server)");
            Console.ResetColor();
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"[MCP] Failed to list tools: {ex.Message}");
            Console.ResetColor();
        }
    }

    string toolCatalogHint = BuildToolCatalogHint(tools);

    var chatOptions = new ChatOptions
    {
        Tools = tools,
        Instructions =
            $"""
            You are a helpful assistant. Use tools when they help.
            If RAG context snippets appear in the conversation, prefer them for factual answers and cite the source name.
            Be concise unless the user asks for detail.

            When the user asks what tools you have or to list tools, enumerate EVERY registered tool below by name and short description — do not omit MCP or remote tools.

            Registered tools in this session ({tools.Count} total): {toolCatalogHint}
            """,
    };

    ChatClientAgentOptions agentOptions = new() { ChatOptions = chatOptions };

    if (ragEnabled)
    {
        TextSearchProviderOptions searchOpts = new()
        {
            SearchTime = TextSearchProviderOptions.TextSearchBehavior.BeforeAIInvoke,
            RecentMessageMemoryLimit = 8,
        };
        agentOptions.AIContextProviders = [new TextSearchProvider(MockHrSearchAsync, searchOpts)];
    }

    ChatClientAgent innerAgent = pipeline.AsAIAgent(agentOptions);
    AIAgent agent = toolConsoleLog
        ? innerAgent.AsBuilder().Use(LogToolInvocationToConsole).Build()
        : innerAgent;

    PrintBanner(endpoint, modelName, ragEnabled, mcpUrl, useStreaming, tools.Count);

    AgentSession session = await agent.CreateSessionAsync();

    while (true)
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.Write("You: ");
        Console.ResetColor();

        var input = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(input) || input.Equals("quit", StringComparison.OrdinalIgnoreCase))
        {
            break;
        }

        if (input.Equals("!tools", StringComparison.OrdinalIgnoreCase)
            || input.Equals("/tools", StringComparison.OrdinalIgnoreCase))
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("Registered tools (runtime):");
            Console.ResetColor();
            PrintToolList(tools);
            Console.WriteLine();
            continue;
        }

        try
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("Assistant: ");
            Console.ResetColor();

            if (useStreaming)
            {
                await foreach (var update in agent.RunStreamingAsync(input, session))
                {
                    if (update.Text is { Length: > 0 } chunk)
                    {
                        Console.Write(chunk);
                    }
                }

                Console.WriteLine();
            }
            else
            {
                var response = await agent.RunAsync(input, session);
                Console.WriteLine(string.IsNullOrEmpty(response.Text) ? "(no text)" : response.Text);
            }
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Error: {ex.Message}");
            Console.ResetColor();
        }

        Console.WriteLine();
    }
}

static string BuildToolCatalogHint(IReadOnlyList<AITool> tools)
{
    return string.Join("; ", tools.Select(ToolOneLiner));
}

static string ToolOneLiner(AITool t)
{
    if (t is AIFunction f)
    {
        string d = f.Description;
        if (d.Length > 120)
        {
            d = d[..117] + "...";
        }

        return $"{f.Name} — {d}";
    }

    return t.ToString() ?? "?";
}

static void PrintToolList(IReadOnlyList<AITool> tools)
{
    foreach (var t in tools)
    {
        if (t is AIFunction f)
        {
            Console.WriteLine($"  • {f.Name}: {f.Description}");
        }
        else
        {
            Console.WriteLine($"  • {t}");
        }
    }
}

static async ValueTask<object?> LogToolInvocationToConsole(
    AIAgent _,
    FunctionInvocationContext context,
    Func<FunctionInvocationContext, CancellationToken, ValueTask<object?>> next,
    CancellationToken cancellationToken)
{
    string name = context.Function?.Name
        ?? context.CallContent?.Name
        ?? "?";
    Console.ForegroundColor = ConsoleColor.DarkYellow;
    Console.WriteLine($"  >>> [TOOL] {name} {FormatToolArgumentsJson(context)}");
    Console.ResetColor();
    return await next(context, cancellationToken);
}

static string FormatToolArgumentsJson(FunctionInvocationContext context)
{
    try
    {
        if (context.Arguments is null || context.Arguments.Count == 0)
        {
            return "()";
        }

        var dict = new Dictionary<string, object?>(context.Arguments.Count);
        foreach (var kv in context.Arguments)
        {
            dict[kv.Key] = kv.Value;
        }

        string json = JsonSerializer.Serialize(dict);
        const int maxLen = 2048;
        if (json.Length > maxLen)
        {
            return json[..(maxLen - 3)] + "...";
        }

        return json;
    }
    catch
    {
        return "(args)";
    }
}

static bool IsEnvDisabled(string name)
{
    var v = Environment.GetEnvironmentVariable(name);
    if (string.IsNullOrWhiteSpace(v))
    {
        return false;
    }

    return v.Equals("false", StringComparison.OrdinalIgnoreCase)
           || v.Equals("0", StringComparison.OrdinalIgnoreCase)
           || v.Equals("off", StringComparison.OrdinalIgnoreCase);
}

static List<AITool> CreateLocalDemoTools()
{
    var weatherTool = AIFunctionFactory.Create(
        (string city) =>
            $$"""{"city":"{{city}}","temperature":"22°C","condition":"Partly cloudy","humidity":"45%"}""",
        name: "get_weather",
        description: "Gets the current weather for a given city.");

    var searchTool = AIFunctionFactory.Create(
        (string query, int top_k = 3) =>
            $$"""{"hits":[{"title":"Leave Policy 2025","snippet":"Annual leave is 14 working days. Sick leave is 10 days."},{"title":"Remote Work Guide","snippet":"Employees may work remote up to 3 days per week."}]}""",
        name: "search_documents",
        description: "Searches the internal knowledge base for documents matching the query.");

    var calculatorTool = AIFunctionFactory.Create(
        (string expression) =>
            $$"""{"expression":"{{expression}}","result":42}""",
        name: "calculate",
        description: "Evaluates a mathematical expression and returns the result.");

    return [weatherTool, searchTool, calculatorTool];
}

static Task<IEnumerable<TextSearchProvider.TextSearchResult>> MockHrSearchAsync(string query, CancellationToken ct)
{
    List<TextSearchProvider.TextSearchResult> results = [];

    if (query.Contains("return", StringComparison.OrdinalIgnoreCase)
        || query.Contains("refund", StringComparison.OrdinalIgnoreCase))
    {
        results.Add(new TextSearchProvider.TextSearchResult
        {
            SourceName = "Contoso HR — Returns",
            SourceLink = "https://contoso.example/policies/returns",
            Text = "Customers may return any item within 30 days of delivery in original packaging.",
        });
    }

    if (query.Contains("leave", StringComparison.OrdinalIgnoreCase)
        || query.Contains("vacation", StringComparison.OrdinalIgnoreCase)
        || query.Contains("policy", StringComparison.OrdinalIgnoreCase))
    {
        results.Add(new TextSearchProvider.TextSearchResult
        {
            SourceName = "Contoso HR — Leave",
            SourceLink = "https://contoso.example/policies/leave",
            Text = "Annual leave is 14 working days. Sick leave is 10 days.",
        });
    }

    if (query.Contains("remote", StringComparison.OrdinalIgnoreCase))
    {
        results.Add(new TextSearchProvider.TextSearchResult
        {
            SourceName = "Contoso HR — Remote",
            SourceLink = "https://contoso.example/policies/remote",
            Text = "Employees may work remotely up to 3 days per week unless otherwise agreed.",
        });
    }

    return Task.FromResult<IEnumerable<TextSearchProvider.TextSearchResult>>(results);
}

static void PrintBanner(string endpoint, string model, bool rag, string? mcp, bool streaming, int toolCount)
{
    Console.WriteLine("=== Nous unified chat (ConsoleTest) ===");
    Console.WriteLine($"LLM: {model} @ {endpoint}");
    Console.WriteLine($"Tools registered: {toolCount} — type !tools to list without calling the model.");
    Console.WriteLine($"RAG (mock HR): {(rag ? "on" : "off")} — set ENABLE_RAG=false to disable");
    Console.WriteLine($"MCP: {(string.IsNullOrWhiteSpace(mcp) ? "off (set MCP_ENDPOINT to enable)" : mcp)}");
    Console.WriteLine($"Mode: {(streaming ? "STREAMING (--stream)" : "non-streaming")}");
    Console.WriteLine("Try: weather, HR policy, CopilotKit docs (with MCP), or '!tools'.");
    Console.WriteLine();
}
