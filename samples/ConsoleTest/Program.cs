// Copyright (c) Microsoft. All rights reserved.

// Standalone interactive console test for NousToolCallingChatClient with an OpenAI-compatible endpoint.
// Uses FunctionInvokingChatClient from Microsoft.Extensions.AI for the automatic tool invocation loop.
// Configure your endpoint via environment variables: OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY.

using System.ClientModel;
using NousToolCalling;
using Microsoft.Extensions.AI;
using OpenAI;

var endpoint = Environment.GetEnvironmentVariable("OPENAI_ENDPOINT")
    ?? "https://api.openai.com/v1";

var modelName = Environment.GetEnvironmentVariable("OPENAI_MODEL")
    ?? "gpt-4o-mini";

var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
    ?? throw new InvalidOperationException("Set the OPENAI_API_KEY environment variable.");

// Pipeline: FunctionInvokingChatClient (outermost) → NousToolCalling → OpenAI leaf
//
// Builder order: first Use() = outermost layer.
// FIC must be outermost so it sees FunctionCallContent from NousToolCalling.
// NousToolCalling must be innermost (closest to OpenAI leaf) so it can inject
// tools into the system prompt and parse <tool_call> XML from responses.

IChatClient leafClient = new OpenAIClient(
        new ApiKeyCredential(apiKey),
        new OpenAIClientOptions { Endpoint = new Uri(endpoint) })
    .GetChatClient(modelName)
    .AsIChatClient();

IChatClient pipeline = leafClient
    .AsBuilder()
    .UseFunctionInvocation()
    .UseNousToolCalling(new NousToolCallingOptions
    {
        StrictThinkMode = true,
        PreserveThinkBlocks = true,
        ToolCallIdPrefix = "nous",
    })
    .Build();

var weatherTool = AIFunctionFactory.Create(
    (string city) =>
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"  >>> [TOOL] get_weather(city: \"{city}\")");
        Console.ResetColor();
        return $$"""{"city":"{{city}}","temperature":"22°C","condition":"Partly cloudy","humidity":"45%"}""";
    },
    name: "get_weather",
    description: "Gets the current weather for a given city. Returns temperature, condition, and humidity.");

var searchTool = AIFunctionFactory.Create(
    (string query, int top_k = 3) =>
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"  >>> [TOOL] search_documents(query: \"{query}\", top_k: {top_k})");
        Console.ResetColor();
        return $$"""{"hits":[{"title":"Leave Policy 2025","snippet":"Annual leave is 14 working days. Sick leave is 10 days."},{"title":"Remote Work Guide","snippet":"Employees may work remote up to 3 days per week."}]}""";
    },
    name: "search_documents",
    description: "Searches the internal knowledge base for documents matching the query. Returns top_k results with title and snippet.");

var calculatorTool = AIFunctionFactory.Create(
    (string expression) =>
    {
        Console.ForegroundColor = ConsoleColor.DarkYellow;
        Console.WriteLine($"  >>> [TOOL] calculate(expression: \"{expression}\")");
        Console.ResetColor();
        return $$"""{"expression":"{{expression}}","result":42}""";
    },
    name: "calculate",
    description: "Evaluates a mathematical expression and returns the result.");

var tools = new List<AITool> { weatherTool, searchTool, calculatorTool };

var chatOptions = new ChatOptions
{
    Tools = tools,
    Instructions = "You are a helpful assistant. Use tools when needed to answer questions. Be concise.",
};

var useStreaming = args.Contains("--stream", StringComparer.OrdinalIgnoreCase);

Console.WriteLine("=== Nous Tool Calling Test ===");
Console.WriteLine($"Mode: {(useStreaming ? "STREAMING" : "NON-STREAMING")}  (pass --stream to toggle)");
Console.WriteLine("Type a message (or 'quit' to exit).");
Console.WriteLine("Example prompts:");
Console.WriteLine("  - What's the weather in Ankara?");
Console.WriteLine("  - Search for 'leave policy' and tell me the annual leave days.");
Console.WriteLine("  - What's 2+2? Also check the weather in Istanbul.");
Console.WriteLine();

var history = new List<ChatMessage>();

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

    history.Add(new ChatMessage(ChatRole.User, input));

    try
    {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write("Assistant: ");
        Console.ResetColor();

        if (useStreaming)
        {
            var streamResponse = pipeline.GetStreamingResponseAsync(history, chatOptions);
            await foreach (var update in streamResponse)
            {
                if (update.Text is { Length: > 0 } chunk)
                {
                    Console.Write(chunk);
                }
            }

            Console.WriteLine();

            var completed = await streamResponse.ToChatResponseAsync();
            history.AddRange(completed.Messages);
        }
        else
        {
            var response = await pipeline.GetResponseAsync(history, chatOptions);

            var assistantText = response.ToString();
            Console.WriteLine(string.IsNullOrEmpty(assistantText) ? "(no text)" : assistantText);

            history.AddRange(response.Messages);
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
