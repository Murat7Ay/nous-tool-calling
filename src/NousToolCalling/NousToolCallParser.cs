// Copyright (c) Murat Ay. Licensed under the MIT License.

using System.Text.Json;

namespace NousToolCalling;

/// <summary>
/// Represents a completed tool call extracted from assistant content.
/// </summary>
/// <param name="Name">The function name.</param>
/// <param name="Arguments">The parsed arguments dictionary.</param>
/// <param name="Ordinal">The 1-based position of this tool call in the assistant message.</param>
public readonly record struct NousCompletedToolCall(
    string Name,
    IDictionary<string, object?> Arguments,
    int Ordinal);

/// <summary>
/// The result of parsing assistant content for Nous-style tool calls.
/// </summary>
/// <param name="CompletedCalls">Tool calls that are complete and safe to execute.</param>
/// <param name="Text">Non-tool-call text content from the message.</param>
/// <param name="ThinkContent">The content inside <c>&lt;think&gt;...&lt;/think&gt;</c> blocks, if any.</param>
/// <param name="ThinkBlockOpen">Whether a <c>&lt;think&gt;</c> block is still unclosed.</param>
public readonly record struct NousParseResult(
    IReadOnlyList<NousCompletedToolCall> CompletedCalls,
    string Text,
    string? ThinkContent,
    bool ThinkBlockOpen);

/// <summary>
/// Parses assistant content for <c>&lt;tool_call&gt;</c> blocks in the Nous/Hermes JSON format or the GLM XML format.
/// This parser is stateless and operates on the full (possibly accumulated) content string.
/// </summary>
public static class NousToolCallParser
{
    private const string ThinkOpen = "<think>";
    private const string ThinkClose = "</think>";
    private const string ToolCallOpen = "<tool_call>";
    private const string ToolCallClose = "</tool_call>";

    /// <summary>
    /// Parses the raw assistant content and extracts completed tool calls and text.
    /// </summary>
    /// <param name="raw">The full assistant content string.</param>
    /// <param name="strictThink">
    /// When <see langword="true"/>, if a <c>&lt;think&gt;</c> block is open and unclosed,
    /// all tool-call parsing is deferred and the content is returned as text only.
    /// </param>
    /// <returns>A <see cref="NousParseResult"/> containing completed calls and text.</returns>
    public static NousParseResult Parse(string? raw, bool strictThink)
        => Parse(raw, strictThink, NousToolCallFormat.Hermes, null);

    /// <summary>
    /// Parses the raw assistant content in the given tool-call format and extracts completed tool calls and text.
    /// </summary>
    /// <param name="raw">The full assistant content string.</param>
    /// <param name="strictThink">
    /// When <see langword="true"/>, if a <c>&lt;think&gt;</c> block is open and unclosed,
    /// all tool-call parsing is deferred and the content is returned as text only.
    /// </param>
    /// <param name="format">The tool-call format to accept.</param>
    /// <param name="toolSchemas">
    /// Tool name to parameters JSON schema. Used by the GLM format to convert argument text to typed values.
    /// </param>
    /// <returns>A <see cref="NousParseResult"/> containing completed calls and text.</returns>
    public static NousParseResult Parse(
        string? raw,
        bool strictThink,
        NousToolCallFormat format,
        IReadOnlyDictionary<string, JsonElement>? toolSchemas)
    {
        if (string.IsNullOrEmpty(raw))
        {
            return new NousParseResult([], string.Empty, null, false);
        }

        var work = raw;
        string? thinkContent = null;
        bool thinkOpen = false;

        var thinkOpenIdx = work.IndexOf(ThinkOpen, StringComparison.Ordinal);
        var loneCloseIdx = thinkOpenIdx < 0 && format != NousToolCallFormat.Hermes
            ? work.IndexOf(ThinkClose, StringComparison.Ordinal)
            : -1;
        if (loneCloseIdx >= 0)
        {
            // GLM-4.7+ ends the prompt with <think>, so the output may carry only the closing tag:
            // everything before it is reasoning.
            thinkContent = work[..loneCloseIdx].Trim();
            work = work[(loneCloseIdx + ThinkClose.Length)..];
        }
        else if (thinkOpenIdx >= 0)
        {
            var thinkCloseIdx = work.IndexOf(ThinkClose, thinkOpenIdx, StringComparison.Ordinal);
            if (thinkCloseIdx < 0)
            {
                thinkOpen = true;
                if (strictThink)
                {
                    return new NousParseResult([], work, null, true);
                }
            }
            else
            {
                var thinkStart = thinkOpenIdx + ThinkOpen.Length;
                thinkContent = work[thinkStart..thinkCloseIdx].Trim();
                var afterThink = thinkCloseIdx + ThinkClose.Length;
                work = string.Concat(work.AsSpan(0, thinkOpenIdx), work.AsSpan(afterThink));
            }
        }

        var completedCalls = new List<NousCompletedToolCall>();
        var textParts = new List<string>();
        var ordinal = 1;
        var searchStart = 0;

        while (searchStart < work.Length)
        {
            var openIdx = work.IndexOf(ToolCallOpen, searchStart, StringComparison.Ordinal);
            if (openIdx < 0)
            {
                var remaining = work[searchStart..].Trim();
                if (remaining.Length > 0)
                {
                    textParts.Add(remaining);
                }

                break;
            }

            var before = work[searchStart..openIdx].Trim();
            if (before.Length > 0)
            {
                textParts.Add(before);
            }

            var contentStart = openIdx + ToolCallOpen.Length;
            var closeIdx = work.IndexOf(ToolCallClose, contentStart, StringComparison.Ordinal);

            if (closeIdx < 0)
            {
                // Some models omit the closing tag on the last call. Accept it if the rest is a
                // complete call; otherwise (e.g. truncated output) keep it as text so nothing is lost.
                var tail = work[contentStart..].Trim();
                // GLM calls always carry the closing tag (vLLM requires it too), so only JSON calls are accepted here.
                var tailCall = format == NousToolCallFormat.Glm || (format == NousToolCallFormat.Auto && !tail.StartsWith('{'))
                    ? null
                    : TryParseToolCallJson(tail);
                if (tailCall is not null)
                {
                    completedCalls.Add(new NousCompletedToolCall(tailCall.Value.Name, tailCall.Value.Arguments, ordinal++));
                }
                else
                {
                    textParts.Add(string.Concat(ToolCallOpen, tail));
                }

                break;
            }

            var jsonText = work[contentStart..closeIdx].Trim();
            var parsed = TryParseToolCallBody(jsonText, format, toolSchemas);
            if (parsed is not null)
            {
                completedCalls.Add(new NousCompletedToolCall(parsed.Value.Name, parsed.Value.Arguments, ordinal++));
            }
            else
            {
                textParts.Add(string.Concat(ToolCallOpen, jsonText, ToolCallClose));
            }

            searchStart = closeIdx + ToolCallClose.Length;
        }

        var text = string.Join("\n", textParts).Trim();
        return new NousParseResult(completedCalls, text, thinkContent, thinkOpen);
    }

    private static (string Name, IDictionary<string, object?> Arguments)? TryParseToolCallBody(
        string body,
        NousToolCallFormat format,
        IReadOnlyDictionary<string, JsonElement>? toolSchemas) => format switch
        {
            NousToolCallFormat.Glm => GlmToolCallParser.TryParse(body, toolSchemas),
            NousToolCallFormat.Auto when !body.StartsWith('{') => GlmToolCallParser.TryParse(body, toolSchemas),
            _ => TryParseToolCallJson(body),
        };

    private static (string Name, IDictionary<string, object?> Arguments)? TryParseToolCallJson(string jsonText)
    {
        try
        {
            using var doc = JsonDocument.Parse(jsonText);
            var root = doc.RootElement;

            if (root.ValueKind != JsonValueKind.Object)
            {
                return null;
            }

            if (!root.TryGetProperty("name", out var nameElement) || nameElement.ValueKind != JsonValueKind.String)
            {
                return null;
            }

            var name = nameElement.GetString();
            if (string.IsNullOrEmpty(name))
            {
                return null;
            }

            var arguments = new Dictionary<string, object?>();

            // "parameters" is a common alias (Llama-style); a missing key means a no-arg call.
            if (!root.TryGetProperty("arguments", out var argsElement) && !root.TryGetProperty("parameters", out argsElement))
            {
                return (name, arguments);
            }

            if (argsElement.ValueKind == JsonValueKind.String)
            {
                // Some models emit the arguments object as a JSON-encoded string.
                using var argsDoc = JsonDocument.Parse(argsElement.GetString()!);
                return argsDoc.RootElement.ValueKind == JsonValueKind.Object
                    ? (name, ToDictionary(argsDoc.RootElement))
                    : null;
            }

            if (argsElement.ValueKind == JsonValueKind.Null)
            {
                return (name, arguments);
            }

            return argsElement.ValueKind == JsonValueKind.Object ? (name, ToDictionary(argsElement)) : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static Dictionary<string, object?> ToDictionary(JsonElement obj)
    {
        var result = new Dictionary<string, object?>();
        foreach (var prop in obj.EnumerateObject())
        {
            result[prop.Name] = ConvertJsonElement(prop.Value);
        }

        return result;
    }

    internal static object? ConvertJsonElement(JsonElement element)
    {
        return element.ValueKind switch
        {
            JsonValueKind.String => element.GetString(),
            JsonValueKind.Number when element.TryGetInt64(out var l) => l,
            JsonValueKind.Number => element.GetDouble(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => null,
            JsonValueKind.Array => element.EnumerateArray().Select(ConvertJsonElement).ToList(),
            JsonValueKind.Object => element.EnumerateObject().ToDictionary(p => p.Name, p => ConvertJsonElement(p.Value)),
            _ => element.GetRawText(),
        };
    }
}
