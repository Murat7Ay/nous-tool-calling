// Copyright (c) Murat Ay. Licensed under the MIT License.

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
    // AIJsonUtilities.DefaultOptions keeps non-ASCII text (e.g. Turkish) readable instead of \uXXXX escapes,
    // which saves tokens and keeps prompts understandable for the model.
    private static readonly JsonSerializerOptions s_compactJson = AIJsonUtilities.DefaultOptions;

    // GLM templates write each tool and each non-string argument with tojson: one line, no indentation.
    private static readonly JsonSerializerOptions s_glmJson = new(AIJsonUtilities.DefaultOptions) { WriteIndented = false };

    /// <summary>
    /// Converts a collection of <see cref="AITool"/> instances into a Nous-format tools instruction
    /// block suitable for injection into a system message.
    /// </summary>
    /// <param name="tools">The tools to describe. Only <see cref="AIFunction"/> instances are included.</param>
    /// <returns>A string containing the full Nous tools instruction block.</returns>
    public static string BuildToolsSection(IEnumerable<AITool> tools)
        => BuildToolsSection(tools, NousToolCallFormat.Hermes);

    /// <summary>
    /// Converts a collection of <see cref="AITool"/> instances into a tools instruction block in the given format.
    /// <see cref="NousToolCallFormat.Glm"/> follows the zai-org GLM-4.7 / GLM-5.x chat templates; any other value
    /// produces the Hermes block.
    /// </summary>
    /// <param name="tools">The tools to describe. Only <see cref="AIFunction"/> instances are included.</param>
    /// <param name="format">The tool-call format the model should answer in.</param>
    /// <returns>A string containing the full tools instruction block.</returns>
    public static string BuildToolsSection(IEnumerable<AITool> tools, NousToolCallFormat format)
    {
        if (format == NousToolCallFormat.Glm)
        {
            return BuildGlmToolsSection(tools);
        }

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
    /// <param name="toolsSection">The tools section produced by <see cref="BuildToolsSection(IEnumerable{AITool})"/>.</param>
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

    /// <summary>
    /// Wraps a tool result string in <c>&lt;tool_response&gt;</c> XML tags in the given format
    /// (GLM-4.7+ templates write no newlines inside the tags).
    /// </summary>
    public static string FormatToolResponse(string resultText, NousToolCallFormat format)
    {
        return format == NousToolCallFormat.Glm ? $"<tool_response>{resultText}</tool_response>" : FormatToolResponse(resultText);
    }

    /// <summary>
    /// Formats a tool call in the given format for inclusion in assistant message content.
    /// GLM calls are written the way the GLM chat templates render them: string values raw, everything else as JSON.
    /// </summary>
    public static string FormatToolCall(string functionName, IDictionary<string, object?>? arguments, NousToolCallFormat format)
    {
        if (format != NousToolCallFormat.Glm)
        {
            return FormatToolCall(functionName, arguments);
        }

        var sb = new StringBuilder("<tool_call>").Append(functionName);
        if (arguments is not null)
        {
            foreach (var (key, value) in arguments)
            {
                var text = value switch
                {
                    string str => str,
                    JsonElement { ValueKind: JsonValueKind.String } je => je.GetString(),
                    _ => JsonSerializer.Serialize(value, s_glmJson),
                };

                sb.Append("<arg_key>").Append(key).Append("</arg_key><arg_value>").Append(text).Append("</arg_value>");
            }
        }

        return sb.Append("</tool_call>").ToString();
    }

    private static string BuildGlmToolsSection(IEnumerable<AITool> tools)
    {
        // GLM-5.x templates list each function object (name, description, parameters) on its own line.
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

            sb.AppendLine(JsonSerializer.Serialize(functionObj, s_glmJson));
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

            For each function call, output the function name and arguments within the following XML format:
            <tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>
            """;
    }
}
