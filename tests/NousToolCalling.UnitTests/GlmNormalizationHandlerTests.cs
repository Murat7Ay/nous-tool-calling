// Copyright (c) Murat Ay. Licensed under the MIT License.

using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json.Nodes;
using Xunit;

namespace NousToolCalling.UnitTests;

public class GlmNormalizationHandlerTests
{
    private const string Url = "http://model.test/v1/chat/completions";

    // ---------------- requests ----------------

    [Fact]
    public async Task Request_DropsReasoningFromHistory_AndEnableThinkingByDefault()
    {
        var fake = new FakeHandler(_ => Json("""{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"""));
        using var http = new HttpClient(new GlmNormalizationHandler(fake));

        await http.PostAsync(Url, JsonBody("""
            {"model":"m","messages":[
              {"role":"user","content":"q"},
              {"role":"assistant","content":"a","reasoning_content":"r1","reasoning":"r2","tool_calls":[{"id":"t","type":"function","function":{"name":"f","arguments":"{}"}}]}
            ],"chat_template_kwargs":{"enable_thinking":false}}
            """));

        var sent = JsonNode.Parse(fake.LastRequestBody!)!.AsObject();
        var assistant = sent["messages"]![1]!.AsObject();
        Assert.False(assistant.ContainsKey("reasoning_content"));
        Assert.False(assistant.ContainsKey("reasoning"));
        Assert.Equal("a", assistant["content"]!.GetValue<string>());
        Assert.NotNull(assistant["tool_calls"]);
        Assert.False(sent.ContainsKey("chat_template_kwargs"));
        Assert.Equal("application/json", fake.LastRequestContentType);
    }

    [Fact]
    public async Task Request_EnableThinking_IsOptIn()
    {
        var fake = new FakeHandler(_ => Json("""{"choices":[]}"""));
        using var http = new HttpClient(new GlmNormalizationHandler(fake, new GlmNormalizationOptions { EnableThinking = false }));

        await http.PostAsync(Url, JsonBody("""{"model":"m","messages":[]}"""));

        Assert.False(JsonNode.Parse(fake.LastRequestBody!)!["chat_template_kwargs"]!["enable_thinking"]!.GetValue<bool>());
    }

    // ---------------- non-streaming ----------------

    [Fact]
    public async Task NonStream_EmptyContent_UsesReasoning()
    {
        var message = await NonStream("""{"role":"assistant","content":"","reasoning":"Merhaba d\u00fcnya"}""");

        Assert.Equal("Merhaba dünya", message["content"]!.GetValue<string>());
        Assert.Equal("Merhaba dünya", message["reasoning_content"]!.GetValue<string>());
    }

    [Fact]
    public async Task NonStream_ContentAndReasoning_KeepsContent_StripsThink()
    {
        var message = await NonStream("""{"role":"assistant","content":"<think>plan</think>\n\nCevap","reasoning_content":null}""");

        Assert.Equal("Cevap", message["content"]!.GetValue<string>());
        Assert.Equal("plan", message["reasoning_content"]!.GetValue<string>());
    }

    [Fact]
    public async Task NonStream_LoneThinkClose_IsStripped()
    {
        var message = await NonStream("""{"role":"assistant","content":"thinking...</think>Answer"}""");

        Assert.Equal("Answer", message["content"]!.GetValue<string>());
        Assert.Equal("thinking...", message["reasoning_content"]!.GetValue<string>());
    }

    [Fact]
    public async Task NonStream_ToolCallsWithEmptyContent_AreLeftAlone()
    {
        var message = await NonStream("""{"role":"assistant","content":null,"reasoning":"call it","tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{\"a\":\"\\u00fc\"}"}}]}""");

        Assert.Null(message["content"]);
        Assert.Equal("{\"a\":\"\\u00fc\"}", message["tool_calls"]![0]!["function"]!["arguments"]!.GetValue<string>());
    }

    [Fact]
    public async Task NonStream_DecodesEscapes_LeavesControlCodes()
    {
        // The server wrote the escapes as text, so on the wire they are double-escaped.
        var message = await NonStream("""{"role":"assistant","content":"G\\u00fczel \\ud83d\\ude00 \\u001b[0m \\\\u00fc"}""");

        Assert.Equal("Güzel 😀 \\u001b[0m \\\\u00fc", message["content"]!.GetValue<string>());
    }

    [Fact]
    public async Task NonStream_CanBeTurnedOff()
    {
        var options = new GlmNormalizationOptions { DecodeUnicodeEscapes = false, ReasoningAsContentFallback = false };
        var message = await NonStream("""{"role":"assistant","content":"","reasoning":"r \\u00fc"}""", options);

        Assert.Equal("", message["content"]!.GetValue<string>());
        Assert.Equal("r \\u00fc", message["reasoning"]!.GetValue<string>());
    }

    // ---------------- streaming ----------------

    [Fact]
    public async Task Stream_EmptyContent_EmitsJoinedReasoningAsContent()
    {
        var events = await Stream(
            Chunk(Delta(reasoning: "Mer")),
            Chunk(Delta(reasoning: "haba")),
            Chunk(Delta(reasoning: " dün")),
            Chunk(Delta(reasoning: "ya"), finish: "stop"),
            "data: [DONE]\n\n");

        Assert.Equal("Merhaba dünya", Content(events));
        Assert.Equal("Merhaba dünya", Reasoning(events));
        Assert.Equal("stop", events.Last(e => e is not null)!["choices"]![0]!["finish_reason"]!.GetValue<string>());
    }

    [Fact]
    public async Task Stream_ContentAndReasoning_ContentStreamsNormally()
    {
        var events = await Stream(
            Chunk(Delta(reasoning: "plan")),
            Chunk(Delta(content: "Cev")),
            Chunk(Delta(content: "ap"), finish: "stop"),
            "data: [DONE]\n\n");

        Assert.Equal("Cevap", Content(events));
    }

    [Fact]
    public async Task Stream_ToolCallsWithEmptyContent_NoFallback()
    {
        var toolDelta = new JsonObject
        {
            ["tool_calls"] = new JsonArray(new JsonObject { ["index"] = 0, ["id"] = "t1", ["type"] = "function", ["function"] = new JsonObject { ["name"] = "f", ["arguments"] = "{}" } }),
        };

        var events = await Stream(
            Chunk(Delta(reasoning: "call f")),
            Chunk(toolDelta),
            Chunk(new JsonObject(), finish: "tool_calls"),
            "data: [DONE]\n\n");

        Assert.Equal(string.Empty, Content(events));
        Assert.Contains(events, e => e?["choices"]?[0]?["delta"]?["tool_calls"] is not null);
    }

    [Fact]
    public async Task Stream_EscapesSplitAcrossDeltas_AreJoined()
    {
        var events = await Stream(
            Chunk(Delta(content: "G\\u00")),
            Chunk(Delta(content: "fczel \\ud83d")),
            Chunk(Delta(content: "\\ud")),
            Chunk(Delta(content: "e00 \\u001b!")),
            Chunk(Delta(reasoning: "d\\u00")),
            Chunk(Delta(reasoning: "fc"), finish: "stop"),
            "data: [DONE]\n\n");

        Assert.Equal("Güzel 😀 \\u001b!", Content(events));
        Assert.Equal("dü", Reasoning(events));
    }

    [Fact]
    public async Task Stream_EndsWithoutFinishReason_FlushesReasoning()
    {
        var events = await Stream(
            Chunk(Delta(reasoning: "only ")),
            Chunk(Delta(reasoning: "thoughts")),
            "data: [DONE]\n\n");

        Assert.Equal("only thoughts", Content(events));
    }

    [Fact]
    public async Task Stream_EndsInsideEscape_EmitsHeldTextAsIs_AndAddsDone()
    {
        var (events, raw) = await StreamRaw(
            Chunk(Delta(content: "abc \\u00")));

        Assert.Equal("abc \\u00", Content(events));
        Assert.EndsWith("data: [DONE]\n\n", raw);
    }

    [Fact]
    public async Task Stream_DropsEmptyChoicesUsageChunk()
    {
        var usage = new JsonObject { ["id"] = "x", ["object"] = "chat.completion.chunk", ["choices"] = new JsonArray(), ["usage"] = new JsonObject { ["total_tokens"] = 5 } };
        var (events, raw) = await StreamRaw(
            Chunk(Delta(content: "hi"), finish: "stop"),
            $"data: {usage.ToJsonString()}\n\n",
            "data: [DONE]\n\n");

        Assert.DoesNotContain("usage", raw);
        Assert.Equal("hi", Content(events));

        var (_, kept) = await StreamRaw(new GlmNormalizationOptions { DropEmptyChoicesChunks = false }, $"data: {usage.ToJsonString()}\n\n");
        Assert.Contains("usage", kept);
    }

    [Fact]
    public async Task Stream_LengthWithEmptyContent_LogsWarning()
    {
        var logger = new ListLogger();
        await StreamRaw(new GlmNormalizationOptions(), logger, Chunk(Delta(reasoning: "long"), finish: "length"), "data: [DONE]\n\n");

        Assert.Contains(logger.Messages, m => m.Contains("max_tokens"));
    }

    [Fact]
    public async Task Stream_UpstreamBreaks_ClosesCleanly()
    {
        var fake = new FakeHandler(_ => Sse(new BreakingStream(Encoding.UTF8.GetBytes(Chunk(Delta(content: "par")) + "data: {\"choi"))));
        using var http = new HttpClient(new GlmNormalizationHandler(fake));

        var raw = await (await http.PostAsync(Url, JsonBody("{}"))).Content.ReadAsStringAsync();

        Assert.Contains("\"content\":\"par\"", raw);
        Assert.EndsWith("data: [DONE]\n\n", raw);
    }

    // ---------------- errors ----------------

    [Fact]
    public async Task ConnectionFailure_ReturnsOpenAiErrorEnvelope()
    {
        var fake = new FakeHandler(_ => throw new HttpRequestException("No connection could be made"));
        using var http = new HttpClient(new GlmNormalizationHandler(fake));

        var response = await http.PostAsync(Url, JsonBody("{}"));

        Assert.Equal(HttpStatusCode.BadGateway, response.StatusCode);
        var error = JsonNode.Parse(await response.Content.ReadAsStringAsync())!["error"]!;
        Assert.Contains("No connection", error["message"]!.GetValue<string>());
    }

    [Fact]
    public async Task BareVllmError_IsWrapped()
    {
        var fake = new FakeHandler(_ => Json("""{"object":"error","message":"bad request","type":"BadRequestError","code":400}""", HttpStatusCode.BadRequest));
        using var http = new HttpClient(new GlmNormalizationHandler(fake));

        var response = await http.PostAsync(Url, JsonBody("{}"));

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Equal("bad request", JsonNode.Parse(await response.Content.ReadAsStringAsync())!["error"]!["message"]!.GetValue<string>());
    }

    [Fact]
    public async Task OtherEndpoints_AreUntouched()
    {
        var fake = new FakeHandler(_ => Json("""{"data":[{"id":"x","reasoning":"r"}]}"""));
        using var http = new HttpClient(new GlmNormalizationHandler(fake));

        var body = await (await http.GetAsync("http://model.test/v1/models")).Content.ReadAsStringAsync();

        Assert.Equal("""{"data":[{"id":"x","reasoning":"r"}]}""", body);
    }

    [Theory]
    [InlineData("plain", "plain")]
    [InlineData("\\u00fc\\u00e7", "üç")]
    [InlineData("\\ud83d\\ude00", "😀")]
    [InlineData("\\ud83d alone", "\\ud83d alone")]
    [InlineData("\\u0000\\u001b", "\\u0000\\u001b")]
    [InlineData("\\\\u00fc", "\\\\u00fc")]
    [InlineData("\\\\\\u00fc", "\\\\ü")]
    [InlineData("\\u00zz", "\\u00zz")]
    public void Decode_Cases(string input, string expected) => Assert.Equal(expected, UnicodeEscapeDecoder.Decode(input));

    // ---------------- helpers ----------------

    private static async Task<JsonObject> NonStream(string message, GlmNormalizationOptions? options = null)
    {
        var body = $$"""{"id":"c","object":"chat.completion","choices":[{"index":0,"message":{{message}},"finish_reason":"stop"}]}""";
        var fake = new FakeHandler(_ => Json(body));
        using var http = new HttpClient(new GlmNormalizationHandler(fake, options));
        var text = await (await http.PostAsync(Url, JsonBody("{}"))).Content.ReadAsStringAsync();
        return JsonNode.Parse(text)!["choices"]![0]!["message"]!.AsObject();
    }

    private static JsonObject Delta(string? content = null, string? reasoning = null)
    {
        var delta = new JsonObject();
        if (reasoning is not null)
        {
            delta["reasoning"] = reasoning;
        }

        if (content is not null)
        {
            delta["content"] = content;
        }

        return delta;
    }

    private static string Chunk(JsonObject delta, string? finish = null)
    {
        var chunk = new JsonObject
        {
            ["id"] = "chatcmpl-1",
            ["object"] = "chat.completion.chunk",
            ["model"] = "glm",
            ["choices"] = new JsonArray(new JsonObject { ["index"] = 0, ["delta"] = delta, ["finish_reason"] = finish }),
        };

        return $"data: {chunk.ToJsonString()}\n\n";
    }

    private static async Task<List<JsonNode?>> Stream(params string[] pieces) => (await StreamRaw(pieces)).Events;

    private static Task<(List<JsonNode?> Events, string Raw)> StreamRaw(params string[] pieces)
        => StreamRaw(new GlmNormalizationOptions(), null, pieces);

    private static Task<(List<JsonNode?> Events, string Raw)> StreamRaw(GlmNormalizationOptions options, params string[] pieces)
        => StreamRaw(options, null, pieces);

    private static async Task<(List<JsonNode?> Events, string Raw)> StreamRaw(GlmNormalizationOptions options, ListLogger? logger, params string[] pieces)
    {
        var fake = new FakeHandler(_ => Sse(new MemoryStream(Encoding.UTF8.GetBytes(string.Concat(pieces)))));
        using var http = new HttpClient(new GlmNormalizationHandler(fake, options, logger));
        var raw = await (await http.PostAsync(Url, JsonBody("{}"))).Content.ReadAsStringAsync();

        var events = raw.Split('\n')
            .Where(l => l.StartsWith("data: ", StringComparison.Ordinal) && l != "data: [DONE]")
            .Select(l => JsonNode.Parse(l[6..]))
            .ToList();
        return (events, raw);
    }

    private static string Content(List<JsonNode?> events) => string.Concat(events.Select(e => e?["choices"]?[0]?["delta"]?["content"]?.GetValue<string>()));

    private static string Reasoning(List<JsonNode?> events) => string.Concat(events.Select(e => e?["choices"]?[0]?["delta"]?["reasoning_content"]?.GetValue<string>()));

    private static StringContent JsonBody(string json) => new(json, Encoding.UTF8, "application/json");

    private static HttpResponseMessage Json(string body, HttpStatusCode status = HttpStatusCode.OK)
        => new(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };

    private static HttpResponseMessage Sse(Stream body)
    {
        var content = new StreamContent(body);
        content.Headers.ContentType = new MediaTypeHeaderValue("text/event-stream");
        return new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
    }

    private sealed class FakeHandler(Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
    {
        public string? LastRequestBody { get; private set; }

        public string? LastRequestContentType { get; private set; }

        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (request.Content is not null)
            {
                LastRequestBody = await request.Content.ReadAsStringAsync(cancellationToken);
                LastRequestContentType = request.Content.Headers.ContentType?.MediaType;
            }

            return respond(request);
        }
    }

    /// <summary>Returns its bytes, then fails like a dropped connection.</summary>
    private sealed class BreakingStream(byte[] data) : MemoryStream(data)
    {
        public override int Read(byte[] buffer, int offset, int count)
        {
            var n = base.Read(buffer, offset, count);
            return n > 0 ? n : throw new IOException("connection reset");
        }

        public override ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            var n = base.Read(buffer.Span);
            return n > 0 ? ValueTask.FromResult(n) : throw new IOException("connection reset");
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
            => ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
    }

    private sealed class ListLogger : Microsoft.Extensions.Logging.ILogger
    {
        public List<string> Messages { get; } = [];

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;

        public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel, Microsoft.Extensions.Logging.EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
            => Messages.Add(formatter(state, exception));
    }
}
