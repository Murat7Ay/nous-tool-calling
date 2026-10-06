// Copyright (c) Murat Ay. Licensed under the MIT License.

using System.Net;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;

namespace NousToolCalling;

/// <summary>
/// An <see cref="HttpMessageHandler"/> that normalizes OpenAI-compatible <c>/chat/completions</c> traffic for
/// reasoning models served by vLLM, such as GLM-4.5+ and GLM-5.x (including GLM-5.3-Flash).
/// </summary>
/// <remarks>
/// <para>
/// The fixes work on the raw JSON, below any SDK, so they apply to the OpenAI .NET SDK, Microsoft.Extensions.AI.OpenAI
/// and any other client that takes an <see cref="HttpClient"/>:
/// </para>
/// <list type="bullet">
/// <item>Requests: removes <c>reasoning</c> / <c>reasoning_content</c> from history messages and controls
/// <c>chat_template_kwargs.enable_thinking</c>.</item>
/// <item>Responses: reads the reasoning from <c>reasoning_content</c> or <c>reasoning</c>, moves <c>&lt;think&gt;</c> blocks
/// out of the content, uses the reasoning as the answer when the content is empty and there are no tool calls, and
/// decodes literal <c>\uXXXX</c> text.</item>
/// <item>Streams: the same per choice (escapes split across chunks are held until complete), drops the usage-only
/// chunk with an empty <c>choices</c> array, and closes the stream cleanly when the upstream breaks off.</item>
/// <item>Errors: connection failures become an HTTP 502 with an OpenAI-style error body; bare vLLM error bodies are
/// wrapped in <c>{"error": …}</c>.</item>
/// </list>
/// </remarks>
public sealed partial class GlmNormalizationHandler : DelegatingHandler
{
    private static readonly string[] s_reasoningFields = ["reasoning_content", "reasoning"];

    private readonly GlmNormalizationOptions _options;
    private readonly ILogger? _logger;

    /// <summary>
    /// Initializes a new instance of the <see cref="GlmNormalizationHandler"/> class. Set <see cref="DelegatingHandler.InnerHandler"/>
    /// before use, or use <see cref="GlmNormalizationHandler(HttpMessageHandler, GlmNormalizationOptions?, ILogger?)"/>.
    /// </summary>
    /// <param name="options">The normalization settings; defaults when <see langword="null"/>.</param>
    /// <param name="logger">Optional logger for warnings (truncated thinking, broken streams, connection failures).</param>
    public GlmNormalizationHandler(GlmNormalizationOptions? options = null, ILogger? logger = null)
    {
        _options = options ?? new GlmNormalizationOptions();
        _logger = logger;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="GlmNormalizationHandler"/> class around <paramref name="innerHandler"/>.
    /// </summary>
    /// <param name="innerHandler">The handler that sends the requests, e.g. a configured <see cref="SocketsHttpHandler"/>.</param>
    /// <param name="options">The normalization settings; defaults when <see langword="null"/>.</param>
    /// <param name="logger">Optional logger for warnings.</param>
    public GlmNormalizationHandler(HttpMessageHandler innerHandler, GlmNormalizationOptions? options = null, ILogger? logger = null)
        : base(innerHandler)
    {
        _options = options ?? new GlmNormalizationOptions();
        _logger = logger;
    }

    [GeneratedRegex(@"<think>(.*?)</think>", RegexOptions.Singleline)]
    private static partial Regex ThinkBlockRegex();

    /// <inheritdoc/>
    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        var isChat = request.Method == HttpMethod.Post
            && request.RequestUri?.AbsolutePath.TrimEnd('/').EndsWith("/chat/completions", StringComparison.OrdinalIgnoreCase) == true;

        if (isChat)
        {
            await RewriteRequestAsync(request, cancellationToken).ConfigureAwait(false);
        }

        HttpResponseMessage response;
        try
        {
            response = await base.SendAsync(request, cancellationToken).ConfigureAwait(false);
        }
        catch (HttpRequestException ex) when (_options.HandleUpstreamErrors && !cancellationToken.IsCancellationRequested)
        {
            _logger?.LogWarning(ex, "Could not reach the model server at {Uri}.", request.RequestUri);
            return CreateErrorResponse(HttpStatusCode.BadGateway, $"Could not reach the model server: {ex.Message}", "upstream_connection_error", request);
        }

        if (!isChat)
        {
            return response;
        }

        if (!response.IsSuccessStatusCode)
        {
            return _options.HandleUpstreamErrors ? await WrapBareErrorAsync(response, cancellationToken).ConfigureAwait(false) : response;
        }

        var mediaType = response.Content.Headers.ContentType?.MediaType;
        if (string.Equals(mediaType, "text/event-stream", StringComparison.OrdinalIgnoreCase))
        {
            var upstream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            var normalized = new StreamContent(new AsyncEnumerableStream(NormalizeStreamAsync(upstream, cancellationToken)));
            CopyHeaders(response.Content, normalized);
            normalized.Headers.ContentLength = null;
            response.Content = normalized;
        }
        else if (mediaType?.Contains("json", StringComparison.OrdinalIgnoreCase) == true)
        {
            var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            var normalized = new StringContent(NormalizeResponseBody(body), Encoding.UTF8);
            CopyHeaders(response.Content, normalized);
            normalized.Headers.ContentLength = null;
            response.Content.Dispose();
            response.Content = normalized;
        }

        return response;
    }

    /// <summary>Applies the request fixes to a chat completions request body.</summary>
    internal string NormalizeRequestBody(string body)
    {
        if (JsonNode.Parse(body) is not JsonObject root)
        {
            return body;
        }

        if (_options.StripReasoningFromHistory && root["messages"] is JsonArray messages)
        {
            foreach (var message in messages.OfType<JsonObject>())
            {
                foreach (var field in s_reasoningFields)
                {
                    message.Remove(field);
                }
            }
        }

        if (_options.EnableThinking is { } enable)
        {
            if (root["chat_template_kwargs"] is not JsonObject kwargs)
            {
                kwargs = new JsonObject();
                root["chat_template_kwargs"] = kwargs;
            }

            kwargs["enable_thinking"] = enable;
        }
        else if (root["chat_template_kwargs"] is JsonObject kwargs)
        {
            kwargs.Remove("enable_thinking");
            if (kwargs.Count == 0)
            {
                root.Remove("chat_template_kwargs");
            }
        }

        return root.ToJsonString();
    }

    /// <summary>Applies the response fixes to a non-streaming chat completion body.</summary>
    internal string NormalizeResponseBody(string body)
    {
        JsonObject? root;
        try
        {
            root = JsonNode.Parse(body) as JsonObject;
        }
        catch (JsonException)
        {
            return body;
        }

        if (root?["choices"] is not JsonArray choices)
        {
            return body;
        }

        try
        {
            foreach (var choice in choices.OfType<JsonObject>())
            {
                if (choice["message"] is JsonObject message)
                {
                    NormalizeMessage(message, choice["finish_reason"]?.GetValueKind() == JsonValueKind.String ? choice["finish_reason"]!.GetValue<string>() : null);
                }
            }

            return root.ToJsonString();
        }
        catch (Exception ex) when (ex is InvalidOperationException or FormatException)
        {
            // A body this code does not understand is passed on unchanged.
            return body;
        }
    }

    private void NormalizeMessage(JsonObject message, string? finishReason)
    {
        string? reasoning = null;
        foreach (var field in s_reasoningFields)
        {
            if (message[field] is JsonValue v && v.TryGetValue<string>(out var text) && text.Length > 0)
            {
                reasoning ??= text;
            }
        }

        var content = message["content"] is JsonValue cv && cv.TryGetValue<string>(out var c) ? c : null;

        if (_options.StripThinkTags && content is not null)
        {
            var (visible, thought) = SplitThink(content);
            if (thought is not null)
            {
                content = visible;
                reasoning = string.IsNullOrEmpty(reasoning) ? thought : reasoning;
            }
        }

        if (_options.DecodeUnicodeEscapes)
        {
            content = content is null ? null : UnicodeEscapeDecoder.Decode(content);
            reasoning = reasoning is null ? null : UnicodeEscapeDecoder.Decode(reasoning);
        }

        var hasToolCalls = message["tool_calls"] is JsonArray { Count: > 0 };
        if (string.IsNullOrWhiteSpace(content) && !hasToolCalls)
        {
            if (finishReason == "length")
            {
                _logger?.LogWarning(
                    "The model hit max_tokens while still thinking and produced no answer (finish_reason=length, empty content). Increase max_tokens.");
            }

            if (_options.ReasoningAsContentFallback && !string.IsNullOrEmpty(reasoning))
            {
                content = reasoning;
            }
        }

        if (message.ContainsKey("content") || content is not null)
        {
            message["content"] = content;
        }

        if (reasoning is not null)
        {
            foreach (var field in s_reasoningFields)
            {
                if (message.ContainsKey(field) || (field == "reasoning_content" && _options.MirrorReasoningContent))
                {
                    message[field] = reasoning;
                }
            }
        }
    }

    /// <summary>
    /// Splits <c>&lt;think&gt;…&lt;/think&gt;</c> (or text before a lone <c>&lt;/think&gt;</c>, written by GLM-4.7+ whose
    /// prompt already opens the block) out of the content. Returns <see langword="null"/> thought when there is none.
    /// </summary>
    private static (string Visible, string? Thought) SplitThink(string content)
    {
        string? thought = null;
        var visible = content;

        const string close = "</think>";
        var closeIdx = visible.IndexOf(close, StringComparison.Ordinal);
        var openIdx = visible.IndexOf("<think>", StringComparison.Ordinal);
        if (closeIdx >= 0 && (openIdx < 0 || openIdx > closeIdx))
        {
            thought = visible[..closeIdx].Trim();
            visible = visible[(closeIdx + close.Length)..];
        }

        var parts = new List<string>();
        if (thought is { Length: > 0 })
        {
            parts.Add(thought);
        }

        var stripped = ThinkBlockRegex().Replace(visible, m =>
        {
            parts.Add(m.Groups[1].Value.Trim());
            return string.Empty;
        });

        if (thought is null && parts.Count == 0)
        {
            return (content, null);
        }

        return (stripped.Trim(), string.Join("\n\n", parts.Where(p => p.Length > 0)));
    }

    private async Task RewriteRequestAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        if (request.Content is null
            || request.Content.Headers.ContentType?.MediaType?.Contains("json", StringComparison.OrdinalIgnoreCase) != true)
        {
            return;
        }

        var body = await request.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        string rewritten;
        try
        {
            rewritten = NormalizeRequestBody(body);
        }
        catch (Exception ex) when (ex is JsonException or InvalidOperationException)
        {
            return;
        }

        var content = new StringContent(rewritten, Encoding.UTF8);
        CopyHeaders(request.Content, content);
        content.Headers.ContentLength = null;
        request.Content = content;
    }

    private async IAsyncEnumerable<ReadOnlyMemory<byte>> NormalizeStreamAsync(
        Stream upstream,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var normalizer = new GlmStreamNormalizer(_options, _logger);
        using var reader = new StreamReader(upstream, Encoding.UTF8);
        while (true)
        {
            string? line;
            try
            {
                line = await reader.ReadLineAsync(cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (_options.HandleUpstreamErrors && ex is IOException or HttpRequestException
                && !cancellationToken.IsCancellationRequested)
            {
                _logger?.LogWarning(ex, "The model server stream broke off; closing the response with what arrived.");
                break;
            }

            if (line is null)
            {
                break;
            }

            var output = normalizer.ProcessLine(line);
            if (output.Length > 0)
            {
                yield return Encoding.UTF8.GetBytes(output);
            }
        }

        yield return Encoding.UTF8.GetBytes(normalizer.Complete());
    }

    private static async Task<HttpResponseMessage> WrapBareErrorAsync(HttpResponseMessage response, CancellationToken cancellationToken)
    {
        var body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        var wrapped = body;
        try
        {
            if (JsonNode.Parse(body) is JsonObject root && root["error"] is null
                && root["object"]?.GetValueKind() == JsonValueKind.String && root["object"]!.GetValue<string>() == "error")
            {
                wrapped = new JsonObject { ["error"] = root.DeepClone() }.ToJsonString();
            }
        }
        catch (JsonException)
        {
        }

        var content = new StringContent(wrapped, Encoding.UTF8);
        CopyHeaders(response.Content, content);
        content.Headers.ContentLength = null;
        response.Content.Dispose();
        response.Content = content;
        return response;
    }

    private static HttpResponseMessage CreateErrorResponse(HttpStatusCode status, string message, string type, HttpRequestMessage request)
    {
        var body = new JsonObject
        {
            ["error"] = new JsonObject
            {
                ["message"] = message,
                ["type"] = type,
                ["param"] = null,
                ["code"] = (int)status,
            },
        };

        return new HttpResponseMessage(status)
        {
            Content = new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json"),
            RequestMessage = request,
        };
    }

    private static void CopyHeaders(HttpContent from, HttpContent to)
    {
        foreach (var header in from.Headers)
        {
            if (!header.Key.Equals("Content-Length", StringComparison.OrdinalIgnoreCase))
            {
                to.Headers.Remove(header.Key);
                to.Headers.TryAddWithoutValidation(header.Key, header.Value);
            }
        }
    }

    /// <summary>A read-only stream over asynchronously produced byte chunks.</summary>
    private sealed class AsyncEnumerableStream(IAsyncEnumerable<ReadOnlyMemory<byte>> source) : Stream
    {
        private readonly IAsyncEnumerator<ReadOnlyMemory<byte>> _enumerator = source.GetAsyncEnumerator();
        private ReadOnlyMemory<byte> _current;
        private bool _completed;

        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => throw new NotSupportedException();

        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            while (_current.IsEmpty)
            {
                if (_completed || !await _enumerator.MoveNextAsync().ConfigureAwait(false))
                {
                    _completed = true;
                    return 0;
                }

                _current = _enumerator.Current;
            }

            var count = Math.Min(buffer.Length, _current.Length);
            _current[..count].CopyTo(buffer);
            _current = _current[count..];
            return count;
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
            => ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();

        public override int Read(byte[] buffer, int offset, int count)
            => ReadAsync(buffer.AsMemory(offset, count)).AsTask().GetAwaiter().GetResult();

        public override void Flush()
        {
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }

            base.Dispose(disposing);
        }

        public override async ValueTask DisposeAsync()
        {
            await _enumerator.DisposeAsync().ConfigureAwait(false);
            await base.DisposeAsync().ConfigureAwait(false);
        }
    }
}
