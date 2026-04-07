// RAG via Microsoft.Agents.AI TextSearchProvider (MessageAIContextProvider) + Nous ChatClientAgent.
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

TextSearchProviderOptions searchOpts = new()
{
    SearchTime = TextSearchProviderOptions.TextSearchBehavior.BeforeAIInvoke,
    RecentMessageMemoryLimit = 6,
};

ChatClientAgent agent = pipeline.AsAIAgent(new ChatClientAgentOptions
{
    ChatOptions = new ChatOptions
    {
        Instructions =
            "You are a support specialist. Answer using the injected context snippets and cite the source name when possible.",
    },
    AIContextProviders = [new TextSearchProvider(MockSearchAsync, searchOpts)],
});

Console.WriteLine("--- RAG question (returns policy snippet) ---");
var answer = await agent.RunAsync("What is the return window for Contoso Outdoors?");
Console.WriteLine(answer);

static Task<IEnumerable<TextSearchProvider.TextSearchResult>> MockSearchAsync(string query, CancellationToken ct)
{
    List<TextSearchProvider.TextSearchResult> results = [];
    if (query.Contains("return", StringComparison.OrdinalIgnoreCase))
    {
        results.Add(new TextSearchProvider.TextSearchResult
        {
            SourceName = "Return Policy",
            SourceLink = "https://contoso.example/policies/returns",
            Text = "Customers may return any item within 30 days of delivery in original packaging.",
        });
    }

    return Task.FromResult<IEnumerable<TextSearchProvider.TextSearchResult>>(results);
}
