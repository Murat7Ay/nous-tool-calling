// Copyright (c) Murat Ay. Licensed under the MIT License.

using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Xunit;

namespace NousToolCalling.UnitTests;

public class GlmChatClientTests
{
    private static readonly AIFunction s_weather = AIFunctionFactory.Create(
        (string city, int days) => $"{city}: sunny for {days} days",
        name: "get_weather",
        description: "Gets weather");

    private static readonly NousToolCallingOptions s_glm = new() { ToolCallFormat = NousToolCallFormat.Glm };

    [Fact]
    public async Task Prompt_UsesGlmFormat_AndHistoryIsRenderedLikeTheTemplate()
    {
        List<ChatMessage>? sent = null;
        var inner = new FakeClient((messages, _) =>
        {
            sent = messages.ToList();
            return [new TextContent("ok")];
        });

        var client = new NousToolCallingChatClient(inner, s_glm);
        var history = new List<ChatMessage>
        {
            new(ChatRole.User, "Weather?"),
            new(ChatRole.Assistant, [new FunctionCallContent("c1", "get_weather", new Dictionary<string, object?> { ["city"] = "Rize", ["days"] = 2 })]),
            new(ChatRole.Tool, [new FunctionResultContent("c1", "rainy")]),
        };

        await client.GetResponseAsync(history, new ChatOptions { Tools = [s_weather] });

        var system = sent![0].Text;
        Assert.Contains("<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key>", system);
        Assert.Contains("\"name\":\"get_weather\"", system);
        Assert.DoesNotContain("\"type\":\"function\"", system);
        Assert.Equal("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rize</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>", sent[2].Text);
        Assert.Equal(ChatRole.User, sent[3].Role);
        Assert.Equal("<tool_response>rainy</tool_response>", sent[3].Text);
    }

    [Fact]
    public async Task Hermes_PromptUnchanged_ByDefault()
    {
        List<ChatMessage>? sent = null;
        var inner = new FakeClient((messages, _) =>
        {
            sent = messages.ToList();
            return [new TextContent("ok")];
        });

        await new NousToolCallingChatClient(inner).GetResponseAsync([new ChatMessage(ChatRole.User, "x")], new ChatOptions { Tools = [s_weather] });

        Assert.Equal(NousToolPromptBuilder.BuildToolsSection([s_weather]), sent![0].Text);
    }

    [Fact]
    public async Task NonStreaming_ParsesTypedGlmCall()
    {
        var inner = new FakeClient((_, _) => [new TextContent("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rize</arg_value><arg_key>days</arg_key><arg_value>5</arg_value></tool_call>")]);

        var response = await new NousToolCallingChatClient(inner, s_glm)
            .GetResponseAsync([new ChatMessage(ChatRole.User, "x")], new ChatOptions { Tools = [s_weather] });

        var call = Assert.Single(response.Messages[0].Contents.OfType<FunctionCallContent>());
        Assert.Equal(5L, call.Arguments!["days"]);
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
    }

    [Fact]
    public async Task Streaming_TagsSplitAcrossChunks()
    {
        string[] chunks =
        [
            "Hava durumuna bak", "ıyorum.\n<tool", "_call>get_wea", "ther<arg_k", "ey>city</arg_key><arg_va", "lue>Ri",
            "ze</arg_value><arg_key>days</arg_key><arg_value>3</arg_v", "alue></tool_", "call>\n<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Van</arg_value></tool_call>",
        ];

        var updates = await Stream(chunks, s_glm);

        var text = string.Concat(updates.SelectMany(u => u.Contents).OfType<TextContent>().Select(t => t.Text));
        var calls = updates.SelectMany(u => u.Contents).OfType<FunctionCallContent>().ToList();

        Assert.Equal("Hava durumuna bakıyorum.\n", text);
        Assert.Equal(2, calls.Count);
        Assert.Equal("Rize", calls[0].Arguments!["city"]);
        Assert.Equal(3L, calls[0].Arguments!["days"]);
        Assert.Equal("Van", calls[1].Arguments!["city"]);
        Assert.Equal(ChatFinishReason.ToolCalls, updates[^1].FinishReason);
    }

    [Fact]
    public async Task Streaming_LoneThinkClose_DoesNotWaitForever()
    {
        var updates = await Stream(["Let me think", "</think>", "Answer", " text"], s_glm);

        var text = string.Concat(updates.SelectMany(u => u.Contents).OfType<TextContent>().Select(t => t.Text));
        Assert.Contains("Answer text", text);
        Assert.Empty(updates.SelectMany(u => u.Contents).OfType<FunctionCallContent>());
    }

    [Fact]
    public async Task Auto_UsesGlmPrompt_ForGlmModel()
    {
        List<ChatMessage>? sent = null;
        var inner = new FakeClient((messages, _) =>
        {
            sent = messages.ToList();
            return [new TextContent("<tool_call>get_weather<arg_key>city</arg_key><arg_value>A</arg_value></tool_call>")];
        });

        var client = new NousToolCallingChatClient(inner, new NousToolCallingOptions { ToolCallFormat = NousToolCallFormat.Auto });
        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "x")], new ChatOptions { Tools = [s_weather], ModelId = "zai-org/GLM-5.3-Flash" });

        Assert.Contains("<arg_key>", sent![0].Text);
        Assert.Single(response.Messages[0].Contents.OfType<FunctionCallContent>());

        await client.GetResponseAsync([new ChatMessage(ChatRole.User, "x")], new ChatOptions { Tools = [s_weather], ModelId = "Qwen3-32B" });
        Assert.Contains("<args-json-object>", sent![0].Text);
    }

    private static async Task<List<ChatResponseUpdate>> Stream(string[] chunks, NousToolCallingOptions options)
    {
        var client = new NousToolCallingChatClient(new FakeClient((_, _) => [], chunks), options);
        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "x")], new ChatOptions { Tools = [s_weather] }))
        {
            updates.Add(u);
        }

        return updates;
    }

    private sealed class FakeClient(Func<IEnumerable<ChatMessage>, ChatOptions?, IList<AIContent>> respond, string[]? chunks = null) : IChatClient
    {
        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
            => Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, respond(messages, options))));

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var chunk in chunks ?? [])
            {
                await Task.Yield();
                yield return new ChatResponseUpdate(ChatRole.Assistant, chunk);
            }
        }

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose()
        {
        }
    }
}
