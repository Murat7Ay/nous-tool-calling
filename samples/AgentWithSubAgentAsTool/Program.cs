// One agent exposes another as a tool via AIAgent.AsAIFunction (Microsoft.Agents.AI).
// Env: OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY

using System.ClientModel;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using NousToolCalling.AgentFramework;
using OpenAI;

var endpoint = Environment.GetEnvironmentVariable("OPENAI_ENDPOINT")
    ?? throw new InvalidOperationException("Set OPENAI_ENDPOINT.");
var modelName = Environment.GetEnvironmentVariable("OPENAI_MODEL")
    ?? throw new InvalidOperationException("Set OPENAI_MODEL.");
var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY") ?? "not-needed";

IChatClient leafClient = new OpenAIClient(
        new ApiKeyCredential(apiKey),
        new OpenAIClientOptions { Endpoint = new Uri(endpoint) })
    .GetChatClient(modelName)
    .AsIChatClient();

IChatClient pipeline = NousChatClientPipeline.Create(leafClient);

ChatClientAgent specialist = pipeline.AsAIAgent(
    instructions: "You are a geography specialist. Answer only with city population facts. One short sentence.",
    name: "GeoSpecialist");

AIFunction askSpecialist = specialist.AsAIFunction(new AIFunctionFactoryOptions
{
    Name = "ask_geography_specialist",
    Description = "Delegates a geography question to the specialist agent.",
});

ChatClientAgent coordinator = pipeline.AsAIAgent(
    instructions:
    "You coordinate tasks. For population questions, call ask_geography_specialist with the user's question as the query string.",
    name: "Coordinator",
    tools: [askSpecialist]);

var response = await coordinator.RunAsync("What is the population of Ankara? (approximate is fine)");
Console.WriteLine(response);
