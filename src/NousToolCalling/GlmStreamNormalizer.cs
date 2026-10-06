// Copyright (c) Murat Ay. Licensed under the MIT License.

using System.Text;
using System.Text.Json.Nodes;
using Microsoft.Extensions.Logging;

namespace NousToolCalling;

/// <summary>
/// Rewrites an OpenAI-compatible chat completions SSE stream line by line (see <see cref="GlmNormalizationHandler"/>).
/// </summary>
internal sealed class GlmStreamNormalizer
{
    private static readonly string[] s_reasoningFields = ["reasoning_content", "reasoning"];

    private readonly GlmNormalizationOptions _options;
    private readonly ILogger? _logger;
    private readonly SortedDictionary<int, ChoiceState> _choices = [];
    private JsonObject? _lastChunk;
    private bool _doneSeen;

    public GlmStreamNormalizer(GlmNormalizationOptions options, ILogger? logger)
    {
        _options = options;
        _logger = logger;
    }

    /// <summary>Processes one input line (without its line break) and returns the output text to write.</summary>
    public string ProcessLine(string line)
    {
        if (!line.StartsWith("data:", StringComparison.Ordinal))
        {
            return line + "\n";
        }

        var payload = line.AsSpan(5).Trim().ToString();
        if (payload == "[DONE]")
        {
            var tail = FlushOpenChoices();
            _doneSeen = true;
            return tail + line + "\n";
        }

        JsonObject? chunk;
        try
        {
            chunk = JsonNode.Parse(payload) as JsonObject;
        }
        catch (System.Text.Json.JsonException)
        {
            chunk = null;
        }

        if (chunk is null || chunk["choices"] is not JsonArray choices)
        {
            return line + "\n";
        }

        if (choices.Count == 0)
        {
            // vLLM's last chunk carries only usage; some clients fail indexing choices[0].
            return _options.DropEmptyChoicesChunks ? string.Empty : line + "\n";
        }

        try
        {
            for (var position = 0; position < choices.Count; position++)
            {
                if (choices[position] is JsonObject choice)
                {
                    ProcessChoice(choice, position);
                }
            }

            _lastChunk = chunk;
            return "data: " + chunk.ToJsonString() + "\n";
        }
        catch (Exception ex) when (ex is InvalidOperationException or FormatException or System.Text.Json.JsonException)
        {
            // A chunk this code does not understand (e.g. an unpaired surrogate) is passed on unchanged.
            _logger?.LogDebug(ex, "Passing an unexpected stream chunk through unchanged.");
            return line + "\n";
        }
    }

    /// <summary>
    /// Called when the upstream stream ended (normally or not). Emits anything still held and closes the stream.
    /// </summary>
    public string Complete()
    {
        // A blank line first, in case the upstream broke off before ending its last event.
        var sb = new StringBuilder("\n").Append(FlushOpenChoices());
        if (!_doneSeen)
        {
            sb.Append("data: [DONE]\n\n");
            _doneSeen = true;
        }

        return sb.ToString();
    }

    private void ProcessChoice(JsonObject choice, int position)
    {
        var index = choice["index"] is JsonValue iv && iv.TryGetValue<int>(out var i) ? i : position;
        if (!_choices.TryGetValue(index, out var state))
        {
            state = new ChoiceState();
            _choices[index] = state;
        }

        if (state.Finished)
        {
            // A new round for this index (should not happen); start fresh.
            state = new ChoiceState();
            _choices[index] = state;
        }

        if (choice["delta"] is not JsonObject delta)
        {
            delta = new JsonObject();
            choice["delta"] = delta;
        }

        string? reasoningPiece = null;
        foreach (var field in s_reasoningFields)
        {
            if (delta[field] is JsonValue rv && rv.TryGetValue<string>(out var raw))
            {
                var decoded = Decode(state, field, raw);
                delta[field] = decoded;
                reasoningPiece ??= decoded;
            }
        }

        if (reasoningPiece is not null)
        {
            // Reasoning tokens are word pieces: join them with nothing in between.
            state.Reasoning.Append(reasoningPiece);
            if (_options.MirrorReasoningContent && delta["reasoning_content"] is null)
            {
                delta["reasoning_content"] = reasoningPiece;
            }
        }

        if (delta["content"] is JsonValue cv && cv.TryGetValue<string>(out var content))
        {
            if (!string.IsNullOrWhiteSpace(content))
            {
                state.SawContent = true;
            }

            delta["content"] = Decode(state, "content", content);
        }

        if (delta["tool_calls"] is JsonArray { Count: > 0 })
        {
            state.SawToolCalls = true;
        }

        if (choice["finish_reason"] is JsonValue fv && fv.TryGetValue<string>(out var finishReason))
        {
            var extra = FinishChoice(state, finishReason);
            if (extra.Content.Length > 0)
            {
                delta["content"] = (delta["content"]?.GetValue<string>() ?? string.Empty) + extra.Content;
            }

            if (extra.Reasoning.Length > 0)
            {
                AppendReasoning(delta, extra.Reasoning);
            }
        }
    }

    private (string Content, string Reasoning) FinishChoice(ChoiceState state, string? finishReason)
    {
        state.Finished = true;

        var heldReasoning = state.FlushReasoning();
        state.Reasoning.Append(heldReasoning);
        var content = state.Decoders.TryGetValue("content", out var cd) ? cd.Flush() : string.Empty;
        if (!string.IsNullOrWhiteSpace(content))
        {
            state.SawContent = true;
        }

        if (!state.SawContent && !state.SawToolCalls)
        {
            if (finishReason == "length")
            {
                _logger?.LogWarning(
                    "The model hit max_tokens while still thinking and produced no answer (finish_reason=length, empty content). Increase max_tokens.");
            }

            if (_options.ReasoningAsContentFallback && state.Reasoning.Length > 0)
            {
                content += state.Reasoning.ToString();
            }
        }

        return (content, heldReasoning);
    }

    private string FlushOpenChoices()
    {
        var sb = new StringBuilder();
        foreach (var (index, state) in _choices)
        {
            if (state.Finished)
            {
                continue;
            }

            var (content, reasoning) = FinishChoice(state, null);
            if (content.Length == 0 && reasoning.Length == 0)
            {
                continue;
            }

            var delta = new JsonObject();
            if (reasoning.Length > 0)
            {
                AppendReasoning(delta, reasoning);
            }

            if (content.Length > 0)
            {
                delta["content"] = content;
            }

            var chunk = new JsonObject
            {
                ["id"] = _lastChunk?["id"]?.DeepClone(),
                ["object"] = _lastChunk?["object"]?.DeepClone() ?? "chat.completion.chunk",
                ["created"] = _lastChunk?["created"]?.DeepClone(),
                ["model"] = _lastChunk?["model"]?.DeepClone(),
                ["choices"] = new JsonArray(new JsonObject
                {
                    ["index"] = index,
                    ["delta"] = delta,
                    ["finish_reason"] = null,
                }),
            };

            sb.Append("data: ").Append(chunk.ToJsonString()).Append("\n\n");
        }

        return sb.ToString();
    }

    private void AppendReasoning(JsonObject delta, string text)
    {
        foreach (var field in s_reasoningFields)
        {
            if (delta[field] is not null || (field == "reasoning_content" && _options.MirrorReasoningContent))
            {
                delta[field] = (delta[field]?.GetValue<string>() ?? string.Empty) + text;
            }
        }

        if (delta["reasoning_content"] is null && delta["reasoning"] is null)
        {
            delta["reasoning"] = text;
        }
    }

    private string Decode(ChoiceState state, string field, string text)
    {
        if (!_options.DecodeUnicodeEscapes)
        {
            return text;
        }

        if (!state.Decoders.TryGetValue(field, out var decoder))
        {
            decoder = new StreamingUnicodeEscapeDecoder();
            state.Decoders[field] = decoder;
        }

        return decoder.Push(text);
    }

    private sealed class ChoiceState
    {
        public StringBuilder Reasoning { get; } = new();

        public Dictionary<string, StreamingUnicodeEscapeDecoder> Decoders { get; } = [];

        public bool SawContent { get; set; }

        public bool SawToolCalls { get; set; }

        public bool Finished { get; set; }

        /// <summary>Returns the held reasoning text (one reasoning field only, so it is not counted twice).</summary>
        public string FlushReasoning()
        {
            string? result = null;
            foreach (var field in s_reasoningFields)
            {
                if (Decoders.TryGetValue(field, out var decoder))
                {
                    var rest = decoder.Flush();
                    result ??= rest;
                }
            }

            return result ?? string.Empty;
        }
    }
}
