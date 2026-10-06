// Copyright (c) Murat Ay. Licensed under the MIT License.

using System.Text.Json;
using Xunit;

namespace NousToolCalling.UnitTests;

public class GlmToolCallParserTests
{
    private static readonly IReadOnlyDictionary<string, JsonElement> s_schemas = new Dictionary<string, JsonElement>
    {
        ["get_weather"] = Schema("""{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}"""),
        ["search"] = Schema("""
            {"type":"object","properties":{
              "query":{"type":"string"},
              "limit":{"type":"integer"},
              "score":{"type":"number"},
              "exact":{"type":"boolean"},
              "filters":{"type":"object"},
              "tags":{"type":"array","items":{"type":"string"}},
              "note":{"type":["string","null"]},
              "page":{"type":["integer","null"]}
            }}
            """),
    };

    private static JsonElement Schema(string json) => JsonDocument.Parse(json).RootElement.Clone();

    private static NousParseResult Parse(string raw) =>
        NousToolCallParser.Parse(raw, strictThink: true, NousToolCallFormat.Glm, s_schemas);

    [Fact]
    public void Parses_Glm47_SingleLine()
    {
        var result = Parse("<tool_call>get_weather<arg_key>city</arg_key><arg_value>İstanbul</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call>");

        var call = Assert.Single(result.CompletedCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("İstanbul", call.Arguments["city"]);
        Assert.Equal(3L, call.Arguments["days"]);
        Assert.Equal(string.Empty, result.Text);
    }

    [Fact]
    public void Parses_Glm45_WithNewlines()
    {
        var raw = "\n<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Ankara</arg_value>\n<arg_key>days</arg_key>\n<arg_value>2</arg_value>\n</tool_call>";

        var call = Assert.Single(Parse(raw).CompletedCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Ankara", call.Arguments["city"]);
        Assert.Equal(2L, call.Arguments["days"]);
    }

    [Fact]
    public void Parses_LiteralBackslashN_BetweenKeyAndValue()
    {
        var call = Assert.Single(Parse(@"<tool_call>get_weather<arg_key>city</arg_key>\n<arg_value>Izmir</arg_value></tool_call>").CompletedCalls);
        Assert.Equal("Izmir", call.Arguments["city"]);
    }

    [Theory]
    [InlineData("<tool_call>get_time</tool_call>")]
    [InlineData("<tool_call>get_time\n</tool_call>")]
    [InlineData("<tool_call> get_time </tool_call>")]
    public void Parses_NoArguments(string raw)
    {
        var call = Assert.Single(Parse(raw).CompletedCalls);
        Assert.Equal("get_time", call.Name);
        Assert.Empty(call.Arguments);
    }

    [Fact]
    public void Converts_Values_UsingSchema()
    {
        var raw = "<tool_call>search"
            + "<arg_key>query</arg_key><arg_value> 42 </arg_value>"
            + "<arg_key>limit</arg_key><arg_value>10</arg_value>"
            + "<arg_key>score</arg_key><arg_value>0.75</arg_value>"
            + "<arg_key>exact</arg_key><arg_value>true</arg_value>"
            + "<arg_key>filters</arg_key><arg_value>{\"lang\": \"tr\", \"year\": 2024}</arg_value>"
            + "<arg_key>tags</arg_key><arg_value>[\"a\", \"b\"]</arg_value>"
            + "<arg_key>note</arg_key><arg_value>null</arg_value>"
            + "<arg_key>page</arg_key><arg_value>null</arg_value>"
            + "</tool_call>";

        var args = Assert.Single(Parse(raw).CompletedCalls).Arguments;

        Assert.Equal(" 42 ", args["query"]); // strings stay verbatim
        Assert.Equal(10L, args["limit"]);
        Assert.Equal(0.75, args["score"]);
        Assert.Equal(true, args["exact"]);
        var filters = Assert.IsType<Dictionary<string, object?>>(args["filters"]);
        Assert.Equal("tr", filters["lang"]);
        Assert.Equal(2024L, filters["year"]);
        Assert.Equal(["a", "b"], Assert.IsType<List<object?>>(args["tags"]));
        Assert.Equal("null", args["note"]); // string type wins
        Assert.Null(args["page"]);
    }

    [Fact]
    public void Keeps_String_WhenValueDoesNotFitType_OrSchemaUnknown()
    {
        var result = NousToolCallParser.Parse(
            "<tool_call>search<arg_key>limit</arg_key><arg_value>many</arg_value><arg_key>other</arg_key><arg_value>5</arg_value></tool_call>"
            + "<tool_call>unknown_tool<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",
            strictThink: true, NousToolCallFormat.Glm, s_schemas);

        Assert.Equal(2, result.CompletedCalls.Count);
        Assert.Equal("many", result.CompletedCalls[0].Arguments["limit"]);
        Assert.Equal("5", result.CompletedCalls[0].Arguments["other"]);
        Assert.Equal("1", result.CompletedCalls[1].Arguments["x"]);
    }

    [Fact]
    public void Parses_MultipleCalls_WithTextAround()
    {
        var raw = "Bakıyorum.\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Bursa</arg_value></tool_call>\n"
            + "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Konya</arg_value></tool_call>\nBitti.";

        var result = Parse(raw);

        Assert.Equal(2, result.CompletedCalls.Count);
        Assert.Equal("Bursa", result.CompletedCalls[0].Arguments["city"]);
        Assert.Equal("Konya", result.CompletedCalls[1].Arguments["city"]);
        Assert.Equal([1, 2], result.CompletedCalls.Select(c => c.Ordinal));
        Assert.Equal("Bakıyorum.\nBitti.", result.Text);
    }

    [Fact]
    public void LoneThinkClose_IsReasoning_AndDoesNotBlock()
    {
        // GLM-4.7+ prompts end with <think>, so the output carries only the closing tag.
        var raw = "I should call <tool_call>get_time</tool_call> maybe.</think>Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Van</arg_value></tool_call>";

        var result = Parse(raw);

        var call = Assert.Single(result.CompletedCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Checking.", result.Text);
        Assert.False(result.ThinkBlockOpen);
        Assert.Contains("I should call", result.ThinkContent);
    }

    [Fact]
    public void UnclosedGlmCall_IsKeptAsText()
    {
        var result = Parse("Text <tool_call>get_weather<arg_key>city</arg_key><arg_value>Ist");

        Assert.Empty(result.CompletedCalls);
        Assert.Contains("<tool_call>get_weather", result.Text);
    }

    [Fact]
    public void MalformedBody_IsKeptAsText()
    {
        var result = Parse("<tool_call>get_weather junk<arg_key>city</arg_key><arg_value>X</arg_value></tool_call>");

        Assert.Empty(result.CompletedCalls);
        Assert.Contains("junk", result.Text);
    }

    [Fact]
    public void Auto_AcceptsBothFormats()
    {
        var raw = "<tool_call>{\"name\": \"get_time\", \"arguments\": {}}</tool_call>"
            + "<tool_call>get_weather<arg_key>days</arg_key><arg_value>4</arg_value></tool_call>";

        var result = NousToolCallParser.Parse(raw, strictThink: true, NousToolCallFormat.Auto, s_schemas);

        Assert.Equal(["get_time", "get_weather"], result.CompletedCalls.Select(c => c.Name));
        Assert.Equal(4L, result.CompletedCalls[1].Arguments["days"]);
    }

    [Fact]
    public void Hermes_DoesNotParseGlmBodies()
    {
        var result = NousToolCallParser.Parse("<tool_call>get_time</tool_call>", strictThink: true);

        Assert.Empty(result.CompletedCalls);
    }
}
