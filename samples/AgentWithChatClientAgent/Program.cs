// ChatClientAgent + Nous pipeline (parity with dotnet Agent_With_NousToolCalling sample).
// Env: OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY (same as samples/ConsoleTest)

using System.ClientModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using NousToolCalling.AgentFramework;
using OpenAI;

var endpoint = Environment.GetEnvironmentVariable("OPENAI_ENDPOINT")
    ?? throw new InvalidOperationException("Set OPENAI_ENDPOINT (e.g. http://localhost:8080/v1).");
var modelName = Environment.GetEnvironmentVariable("OPENAI_MODEL")
    ?? throw new InvalidOperationException("Set OPENAI_MODEL (e.g. qwen3-32b).");
var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "not-needed";

IChatClient leafClient = new OpenAIClient(
        new ApiKeyCredential(apiKey),
        new OpenAIClientOptions { Endpoint = new Uri(endpoint) })
    .GetChatClient(modelName)
    .AsIChatClient();

IChatClient pipeline = NousChatClientPipeline.Create(leafClient);

var weatherTool = AIFunctionFactory.Create(
    (string city) =>
    {
        Console.WriteLine($"  [Tool] get_weather({city})");
        return $$"""{"city":"{{city}}","temperature":"22°C","condition":"Partly cloudy"}""";
    },
    name: "get_weather",
    description: "Gets the current weather for a given city.");

var searchTool = AIFunctionFactory.Create(
    (string query, int top_k = 3) =>
    {
        Console.WriteLine($"  [Tool] search_kb({query}, {top_k})");
        return """{"hits":[{"title":"Leave Policy","snippet":"Annual leave is 14 working days..."}]}""";
    },
    name: "search_kb",
    description: "Searches the internal knowledge base for relevant documents.");

AIAgent agent = pipeline.AsAIAgent(
    instructions: "You are a helpful assistant. Answer concisely. Use tools when needed.",
    name: "NousAgent",
    tools: [weatherTool, searchTool]);

Console.WriteLine("--- Single tool call ---");
var response1 = await agent.RunAsync("What's the weather in Ankara?");
Console.WriteLine($"Agent: {response1}\n");

Console.WriteLine("--- Combined ---");
var response2 = await agent.RunAsync("Check the weather in Istanbul and search for 'leave policy' in the KB.");
Console.WriteLine($"Agent: {response2}\n");
