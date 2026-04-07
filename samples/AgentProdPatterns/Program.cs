// Patterns: reusable AgentSession across turns + OpenTelemetryAgent (console exporter for demo).
// Env: OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY
// Optional: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true (avoid in prod with sensitive data)

using System.ClientModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using NousToolCalling.AgentFramework;
using OpenAI;
using OpenTelemetry;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

var endpoint = Environment.GetEnvironmentVariable("OPENAI_ENDPOINT")
    ?? throw new InvalidOperationException("Set OPENAI_ENDPOINT.");
var modelName = Environment.GetEnvironmentVariable("OPENAI_MODEL")
    ?? throw new InvalidOperationException("Set OPENAI_MODEL.");
var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "not-needed";

using var tracerProvider = Sdk.CreateTracerProviderBuilder()
    .SetResourceBuilder(ResourceBuilder.CreateDefault().AddService("nous-agent-demo"))
    .AddConsoleExporter()
    .Build();

IChatClient leafClient = new OpenAIClient(
        new ApiKeyCredential(apiKey),
        new OpenAIClientOptions { Endpoint = new Uri(endpoint) })
    .GetChatClient(modelName)
    .AsIChatClient();

IChatClient pipeline = NousChatClientPipeline.Create(leafClient);

ChatClientAgent inner = pipeline.AsAIAgent(
    instructions: "You are a helpful assistant. Remember the user's stated favorite color for follow-ups.",
    name: "SessionDemo");

using var observable = new OpenTelemetryAgent(inner, sourceName: "nous-tool-calling-prod-patterns");

AIAgent agent = observable;
AgentSession session = await agent.CreateSessionAsync();

Console.WriteLine(await agent.RunAsync("My favorite color is teal.", session));
Console.WriteLine(await agent.RunAsync("What is my favorite color?", session));
