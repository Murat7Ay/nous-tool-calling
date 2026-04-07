// Sequential multi-agent workflow (writer → reviewer) as a single AIAgent host.
// Env: OPENAI_ENDPOINT, OPENAI_MODEL, OPENAI_API_KEY

using System.ClientModel;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
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

AIAgent writer = pipeline.AsAIAgent(
    instructions: "You are an excellent writer. Produce clear, concrete prose. Be concise.",
    name: "Writer",
    description: "Creates and edits written content.");

AIAgent reviewer = pipeline.AsAIAgent(
    instructions: "You are a reviewer. Give brief, actionable feedback. End with a one-sentence improved version.",
    name: "Reviewer",
    description: "Reviews drafts and suggests improvements.");

var workflow = AgentWorkflowBuilder.BuildSequential("writer-reviewer", writer, reviewer);
AIAgent team = workflow.AsAIAgent(name: "Team", description: "Writer then reviewer pipeline.");

var response = await team.RunAsync("Write one sentence explaining what a tool-calling LLM does.");
Console.WriteLine(response);
