// Copyright (c) Murat Ay. Licensed under the MIT License.

using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Xunit;

namespace NousToolCalling.UnitTests;

public class NousToolCallingRobustnessTests
{
    private static readonly AITool s_weatherTool = AIFunctionFactory.Create(
        (string city) => $"Sunny in {city}",
        name: "get_weather",
        description: "Hava durumunu getirir (İstanbul, Ağrı)");

    [Fact]
    public async Task Streaming_ThinkThenToolCall_DoesNotDuplicateText()
    {
        var client = new NousToolCallingChatClient(new StreamingTestClient(
            "<think>", "plan it", "</think>", "Let me ", "check.", "<tool_", "call>{\"name\":\"get_weather\",", "\"arguments\":{\"city\":\"Izmir\"}}</tool_call>"));

        var updates = await ToListAsync(client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "hi")]));

        var text = string.Concat(updates.Select(u => u.Text));
        Assert.Equal(1, CountOccurrences(text, "Let me check."));
        Assert.DoesNotContain("<tool_call>", text);
        var call = Assert.Single(updates.SelectMany(u => u.Contents).OfType<FunctionCallContent>());
        Assert.Equal("Izmir", call.Arguments!["city"]);
    }

    [Fact]
    public async Task Streaming_PreservesUsageAndFinishReason()
    {
        var usage = new UsageContent(new UsageDetails { InputTokenCount = 10, OutputTokenCount = 5 });
        var inner = new StreamingTestClient(
            new ChatResponseUpdate(ChatRole.Assistant, "Hello there"),
            new ChatResponseUpdate { Contents = [usage], FinishReason = ChatFinishReason.Length, ModelId = "qwen3" });

        var updates = await ToListAsync(new NousToolCallingChatClient(inner).GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "hi")]));

        Assert.Contains(updates.SelectMany(u => u.Contents), c => c is UsageContent);
        Assert.Contains(updates, u => u.FinishReason == ChatFinishReason.Length);
        Assert.Contains(updates, u => u.ModelId == "qwen3");
        Assert.Equal("Hello there", string.Concat(updates.Select(u => u.Text)));
    }

    [Fact]
    public async Task Request_ClearsToolModeAndParallelFlag_WhenToolsAreStripped()
    {
        ChatOptions? captured = null;
        var client = new NousToolCallingChatClient(new CapturingClient((_, o) => captured = o));

        await client.GetResponseAsync(
            [new ChatMessage(ChatRole.User, "hi")],
            new ChatOptions { Tools = [s_weatherTool], ToolMode = ChatToolMode.RequireAny, AllowMultipleToolCalls = false });

        Assert.NotNull(captured);
        Assert.Null(captured!.Tools);
        Assert.Null(captured.ToolMode);
        Assert.Null(captured.AllowMultipleToolCalls);
    }

    [Fact]
    public async Task Request_ToolModeNone_DoesNotInjectTools()
    {
        IEnumerable<ChatMessage>? captured = null;
        var client = new NousToolCallingChatClient(new CapturingClient((m, _) => captured = m));

        await client.GetResponseAsync(
            [new ChatMessage(ChatRole.User, "hi")],
            new ChatOptions { Tools = [s_weatherTool], ToolMode = ChatToolMode.None });

        Assert.DoesNotContain(captured!, m => m.Text.Contains("<tools>"));
    }

    [Fact]
    public async Task Request_NonAsciiToolDescription_IsNotEscaped()
    {
        IEnumerable<ChatMessage>? captured = null;
        var client = new NousToolCallingChatClient(new CapturingClient((m, _) => captured = m));

        await client.GetResponseAsync([new ChatMessage(ChatRole.User, "hi")], new ChatOptions { Tools = [s_weatherTool] });

        var system = captured!.First(m => m.Role == ChatRole.System).Text;
        Assert.Contains("İstanbul, Ağrı", system);
    }

    [Fact]
    public async Task Response_CallIdsAreUniqueAcrossTurns()
    {
        const string reply = "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Bursa\"}}</tool_call>";
        var client = new NousToolCallingChatClient(new CapturingClient((_, _) => { }, reply));

        var first = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "a")]);
        var second = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "b")]);

        var id1 = first.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>().Single().CallId;
        var id2 = second.Messages.SelectMany(m => m.Contents).OfType<FunctionCallContent>().Single().CallId;
        Assert.StartsWith("nous_", id1);
        Assert.NotEqual(id1, id2);
    }

    [Fact]
    public void Parser_AcceptsStringifiedArguments()
    {
        var result = NousToolCallParser.Parse(
            "<tool_call>{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Van\\\"}\"}</tool_call>", strictThink: true);

        var call = Assert.Single(result.CompletedCalls);
        Assert.Equal("Van", call.Arguments["city"]);
    }

    [Fact]
    public void Parser_AcceptsMissingArgumentsAsEmpty()
    {
        var result = NousToolCallParser.Parse("<tool_call>{\"name\":\"get_time\"}</tool_call>", strictThink: true);

        var call = Assert.Single(result.CompletedCalls);
        Assert.Empty(call.Arguments);
    }

    [Fact]
    public void Parser_AcceptsMissingClosingTagAtEndOfMessage()
    {
        var result = NousToolCallParser.Parse(
            "Checking.\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Rize\"}}\n", strictThink: true);

        var call = Assert.Single(result.CompletedCalls);
        Assert.Equal("Rize", call.Arguments["city"]);
        Assert.Equal("Checking.", result.Text);
    }

    [Fact]
    public void Parser_TruncatedToolCall_IsKeptAsText()
    {
        var result = NousToolCallParser.Parse("Checking.\n<tool_call>{\"name\":\"get_wea", strictThink: true);

        Assert.Empty(result.CompletedCalls);
        Assert.Contains("<tool_call>{\"name\":\"get_wea", result.Text);
    }

    private static int CountOccurrences(string text, string value)
    {
        int count = 0, idx = 0;
        while ((idx = text.IndexOf(value, idx, StringComparison.Ordinal)) >= 0)
        {
            count++;
            idx += value.Length;
        }

        return count;
    }

    private static async Task<List<T>> ToListAsync<T>(IAsyncEnumerable<T> source)
    {
        var list = new List<T>();
        await foreach (var item in source)
        {
            list.Add(item);
        }

        return list;
    }

    private sealed class StreamingTestClient : IChatClient
    {
        private readonly ChatResponseUpdate[] _updates;

        public StreamingTestClient(params string[] chunks)
            : this(chunks.Select(c => new ChatResponseUpdate(ChatRole.Assistant, c)).ToArray())
        {
        }

        public StreamingTestClient(params ChatResponseUpdate[] updates) => _updates = updates;

        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var update in _updates)
            {
                await Task.Yield();
                yield return update;
            }
        }

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose() { }
    }

    private sealed class CapturingClient : IChatClient
    {
        private readonly Action<IEnumerable<ChatMessage>, ChatOptions?> _capture;
        private readonly string _reply;

        public CapturingClient(Action<IEnumerable<ChatMessage>, ChatOptions?> capture, string reply = "ok")
        {
            _capture = capture;
            _reply = reply;
        }

        public Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
        {
            _capture(messages.ToList(), options);
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, _reply)));
        }

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose() { }
    }
}
