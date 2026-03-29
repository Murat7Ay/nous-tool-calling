// Copyright (c) Microsoft. All rights reserved.

using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace NousToolCalling;

/// <summary>
/// Builds the Nous-style <c>&lt;tools&gt;</c> system prompt section from <see cref="AITool"/> definitions,
/// and formats tool results as <c>&lt;tool_response&gt;</c> blocks.
/// </summary>
public static class NousToolPromptBuilder
{
    private static readonly JsonSerializerOptions s_compactJson = new() { WriteIndented = false };

    /// <summary>
    /// Converts a collection of <see cref="AITool"/> instances into a Nous-format tools instruction
    /// block suitable for injection into a system message.
    /// </summary>
    /// <param name="tools">The tools to describe. Only <see cref="AIFunction"/> instances are included.</param>
    /// <returns>A string containing the full Nous tools instruction block.</returns>
    public static string BuildToolsSection(IEnumerable<AITool> tools)
    {
        var sb = new StringBuilder();

        foreach (var tool in tools)
        {
            if (tool is not AIFunction fn)
            {
                continue;
            }

            var functionObj = new Dictionary<string, object?>
            {
                ["name"] = fn.Name,
                ["description"] = fn.Description,
                ["parameters"] = fn.JsonSchema,
            };

            var line = new Dictionary<string, object?>
            {
                ["type"] = "function",
                ["function"] = functionObj,
            };

            sb.AppendLine(JsonSerializer.Serialize(line, s_compactJson));
        }

        var toolDescriptions = sb.ToString().TrimEnd();

        if (string.IsNullOrEmpty(toolDescriptions))
        {
            return string.Empty;
        }

        return $$"""
            # Tools

            You may call one or more functions to assist with the user query.

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            {{toolDescriptions}}
            </tools>

            For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
            <tool_call>
            {"name": <function-name>, "arguments": <args-json-object>}
            </tool_call>
            """;
    }

    /// <summary>
    /// Merges the Nous tools section into an existing system prompt string.
    /// </summary>
    /// <param name="existingSystem">The existing system instructions, which may be <see langword="null"/> or empty.</param>
    /// <param name="toolsSection">The tools section produced by <see cref="BuildToolsSection"/>.</param>
    /// <returns>The combined system prompt.</returns>
    public static string MergeIntoSystem(string? existingSystem, string toolsSection)
    {
        if (string.IsNullOrWhiteSpace(existingSystem))
        {
            return toolsSection.Trim();
        }

        return $"{existingSystem.Trim()}\n\n{toolsSection.Trim()}";
    }

    /// <summary>
    /// Formats a tool call as a Nous-style <c>&lt;tool_call&gt;</c> XML block for inclusion in assistant message content.
    /// </summary>
    public static string FormatToolCall(string functionName, IDictionary<string, object?>? arguments)
    {
        var inner = new Dictionary<string, object?>
        {
            ["name"] = functionName,
            ["arguments"] = arguments ?? new Dictionary<string, object?>(),
        };

        return $"<tool_call>\n{JsonSerializer.Serialize(inner, s_compactJson)}\n</tool_call>";
    }

    /// <summary>
    /// Wraps a tool result string in <c>&lt;tool_response&gt;</c> XML tags.
    /// </summary>
    public static string FormatToolResponse(string resultText)
    {
        return $"<tool_response>\n{resultText}\n</tool_response>";
    }
}
