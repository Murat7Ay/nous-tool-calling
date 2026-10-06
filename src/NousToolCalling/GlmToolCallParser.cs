// Copyright (c) Murat Ay. Licensed under the MIT License.

using System.Globalization;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace NousToolCalling;

/// <summary>
/// Parses the body of a GLM-style tool call
/// (<c>name&lt;arg_key&gt;k&lt;/arg_key&gt;&lt;arg_value&gt;v&lt;/arg_value&gt;...</c>, the text between
/// <c>&lt;tool_call&gt;</c> and <c>&lt;/tool_call&gt;</c>), following vLLM's <c>glm45</c>/<c>glm47</c> tool parsers.
/// </summary>
/// <remarks>
/// GLM-4.5/4.6 put a newline after the name and between the parts; GLM-4.7 and GLM-5.x write everything
/// on one line. Both are accepted, as are calls without arguments. Argument values are plain text:
/// a value is converted to a number, boolean, object or array when the tool's JSON schema says so,
/// and kept as a string otherwise.
/// </remarks>
public static partial class GlmToolCallParser
{
    // Same as vLLM's glm47 parser: whitespace (or a literal "\n" the model sometimes writes) may separate key and value.
    [GeneratedRegex(@"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>", RegexOptions.Singleline)]
    private static partial Regex ArgRegex();

    /// <summary>
    /// Tries to parse the body of a GLM tool call.
    /// </summary>
    /// <param name="body">The text between <c>&lt;tool_call&gt;</c> and <c>&lt;/tool_call&gt;</c>.</param>
    /// <param name="toolSchemas">Tool name to JSON schema (the <c>parameters</c> object), used to type the values. May be <see langword="null"/>.</param>
    /// <returns>The function name and arguments, or <see langword="null"/> when the body is not a GLM call.</returns>
    public static (string Name, IDictionary<string, object?> Arguments)? TryParse(
        string body,
        IReadOnlyDictionary<string, JsonElement>? toolSchemas)
    {
        var work = body.Trim();
        if (work.Length == 0)
        {
            return null;
        }

        var nameEnd = 0;
        while (nameEnd < work.Length && !char.IsWhiteSpace(work[nameEnd]) && work[nameEnd] != '<')
        {
            nameEnd++;
        }

        var name = work[..nameEnd];
        if (name.Length == 0 || name.IndexOfAny(['{', '}', '"', '>']) >= 0)
        {
            return null;
        }

        JsonElement? properties = null;
        if (toolSchemas is not null && toolSchemas.TryGetValue(name, out var schema)
            && schema.ValueKind == JsonValueKind.Object
            && schema.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object)
        {
            properties = props;
        }

        var arguments = new Dictionary<string, object?>();
        var rest = work[nameEnd..];
        var position = 0;
        foreach (Match match in ArgRegex().Matches(rest))
        {
            if (!IsSeparator(rest.AsSpan(position, match.Index - position)))
            {
                return null;
            }

            var key = match.Groups[1].Value.Trim();
            if (key.Length == 0)
            {
                return null;
            }

            JsonElement? propertySchema = properties is { } p && p.TryGetProperty(key, out var ps) ? ps : null;
            arguments[key] = ConvertValue(match.Groups[2].Value, propertySchema);
            position = match.Index + match.Length;
        }

        return IsSeparator(rest.AsSpan(position)) ? (name, arguments) : null;
    }

    /// <summary>
    /// Converts a GLM argument value (always text) to the type its JSON schema asks for.
    /// Strings stay verbatim; numbers, booleans, objects, arrays and null are parsed; when the schema is
    /// missing or the text does not fit, the raw string is returned.
    /// </summary>
    public static object? ConvertValue(string raw, JsonElement? propertySchema)
    {
        var types = new List<string>();
        if (propertySchema is { ValueKind: JsonValueKind.Object } schema)
        {
            CollectTypes(schema, types);
        }

        if (types.Count == 0)
        {
            return raw;
        }

        var trimmed = raw.Trim();
        var allowsString = types.Contains("string");
        if (allowsString)
        {
            return raw;
        }

        if (trimmed.Equals("null", StringComparison.OrdinalIgnoreCase) || trimmed.Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        foreach (var type in types)
        {
            switch (type)
            {
                case "integer":
                    if (long.TryParse(trimmed, NumberStyles.Integer, CultureInfo.InvariantCulture, out var l))
                    {
                        return l;
                    }

                    break;
                case "number":
                    if (long.TryParse(trimmed, NumberStyles.Integer, CultureInfo.InvariantCulture, out var ln))
                    {
                        return ln;
                    }

                    if (double.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out var d))
                    {
                        return d;
                    }

                    break;
                case "boolean":
                    if (bool.TryParse(trimmed, out var b))
                    {
                        return b;
                    }

                    break;
                case "object":
                case "array":
                    if (trimmed.Length == 0)
                    {
                        return type == "object" ? new Dictionary<string, object?>() : new List<object?>();
                    }

                    if (TryParseJson(trimmed, out var element)
                        && element.ValueKind == (type == "object" ? JsonValueKind.Object : JsonValueKind.Array))
                    {
                        return NousToolCallParser.ConvertJsonElement(element);
                    }

                    break;
            }
        }

        // Typed but the text does not fit: use it as JSON if it is JSON, else keep it as written.
        return TryParseJson(trimmed, out var any) ? NousToolCallParser.ConvertJsonElement(any) : raw;
    }

    private static bool IsSeparator(ReadOnlySpan<char> text)
    {
        // Between arguments GLM writes nothing (4.7+) or newlines (4.5); some outputs carry a literal "\n".
        return text.ToString().Replace("\\n", string.Empty, StringComparison.Ordinal).AsSpan().IsWhiteSpace();
    }

    private static void CollectTypes(JsonElement schema, List<string> types)
    {
        if (schema.TryGetProperty("type", out var type))
        {
            if (type.ValueKind == JsonValueKind.String)
            {
                AddType(type.GetString(), types);
            }
            else if (type.ValueKind == JsonValueKind.Array)
            {
                foreach (var t in type.EnumerateArray())
                {
                    if (t.ValueKind == JsonValueKind.String)
                    {
                        AddType(t.GetString(), types);
                    }
                }
            }
        }

        foreach (var keyword in new[] { "anyOf", "oneOf" })
        {
            if (schema.TryGetProperty(keyword, out var options) && options.ValueKind == JsonValueKind.Array)
            {
                foreach (var option in options.EnumerateArray())
                {
                    if (option.ValueKind == JsonValueKind.Object)
                    {
                        CollectTypes(option, types);
                    }
                }
            }
        }

        if (types.Count == 0 && schema.TryGetProperty("enum", out var values) && values.ValueKind == JsonValueKind.Array
            && values.EnumerateArray().All(v => v.ValueKind == JsonValueKind.String))
        {
            types.Add("string");
        }
    }

    private static void AddType(string? type, List<string> types)
    {
        var normalized = type?.ToLowerInvariant() switch
        {
            "string" or "str" or "text" or "varchar" or "char" or "enum" => "string",
            "integer" or "int" => "integer",
            "number" or "float" or "double" => "number",
            "boolean" or "bool" or "binary" => "boolean",
            "object" or "dict" or "map" => "object",
            "array" or "arr" or "list" or "sequence" => "array",
            "null" => "null",
            _ => null,
        };

        if (normalized is not null && !types.Contains(normalized))
        {
            types.Add(normalized);
        }
    }

    private static bool TryParseJson(string text, out JsonElement element)
    {
        try
        {
            using var doc = JsonDocument.Parse(text);
            element = doc.RootElement.Clone();
            return true;
        }
        catch (JsonException)
        {
            element = default;
            return false;
        }
    }
}
