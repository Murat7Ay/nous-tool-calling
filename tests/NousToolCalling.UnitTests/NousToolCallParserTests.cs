// Copyright (c) Microsoft. All rights reserved.

using NousToolCalling;
using Xunit;

namespace NousToolCalling.UnitTests;

public class NousToolCallParserTests
{
    [Fact]
    public void Parse_NullContent_ReturnsEmptyResult()
    {
        // Act
        var result = NousToolCallParser.Parse(null, strictThink: true);

        // Assert
        Assert.Empty(result.CompletedCalls);
        Assert.Equal(string.Empty, result.Text);
        Assert.False(result.ThinkBlockOpen);
    }

    [Fact]
    public void Parse_PlainText_ReturnsTextOnly()
    {
        // Arrange
        var content = "This is just a regular response without any tool calls.";

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Empty(result.CompletedCalls);
        Assert.Equal(content, result.Text);
        Assert.False(result.ThinkBlockOpen);
    }

    [Fact]
    public void Parse_SingleToolCall_ExtractsCorrectly()
    {
        // Arrange
        var content = """
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Ankara"}}
            </tool_call>
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Single(result.CompletedCalls);
        var call = result.CompletedCalls[0];
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Ankara", call.Arguments["city"]?.ToString());
        Assert.Equal(1, call.Ordinal);
        Assert.Equal(string.Empty, result.Text);
    }

    [Fact]
    public void Parse_TextBeforeToolCall_PreservesBothParts()
    {
        // Arrange
        var content = """
            Let me check the weather for you.
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Istanbul"}}
            </tool_call>
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Single(result.CompletedCalls);
        Assert.Equal("get_weather", result.CompletedCalls[0].Name);
        Assert.Contains("Let me check the weather", result.Text);
    }

    [Fact]
    public void Parse_ParallelToolCalls_ExtractsBoth()
    {
        // Arrange
        var content = """
            Checking both.
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Ankara"}}
            </tool_call>
            <tool_call>
            {"name": "search_kb", "arguments": {"query": "leave policy", "top_k": 3}}
            </tool_call>
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Equal(2, result.CompletedCalls.Count);
        Assert.Equal("get_weather", result.CompletedCalls[0].Name);
        Assert.Equal(1, result.CompletedCalls[0].Ordinal);
        Assert.Equal("search_kb", result.CompletedCalls[1].Name);
        Assert.Equal(2, result.CompletedCalls[1].Ordinal);
        Assert.Contains("Checking both", result.Text);
    }

    [Fact]
    public void Parse_MalformedJson_TreatsAsText()
    {
        // Arrange
        var content = """
            <tool_call>
            {not valid json}
            </tool_call>
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Empty(result.CompletedCalls);
        Assert.Contains("<tool_call>", result.Text);
    }

    [Fact]
    public void Parse_MissingNameField_TreatsAsText()
    {
        // Arrange
        var content = """
            <tool_call>
            {"arguments": {"city": "Ankara"}}
            </tool_call>
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Empty(result.CompletedCalls);
        Assert.Contains("<tool_call>", result.Text);
    }

    [Fact]
    public void Parse_IncompleteToolCallBlock_NotExtracted()
    {
        // Arrange
        var content = """
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Ankara"}}
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Empty(result.CompletedCalls);
    }

    [Fact]
    public void Parse_ThinkBlockClosed_ExtractsToolCalls()
    {
        // Arrange
        var content = """
            <think>
            I need to look up the weather first.
            </think>
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Ankara"}}
            </tool_call>
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Single(result.CompletedCalls);
        Assert.Equal("get_weather", result.CompletedCalls[0].Name);
        Assert.False(result.ThinkBlockOpen);
        Assert.Equal("I need to look up the weather first.", result.ThinkContent);
    }

    [Fact]
    public void Parse_ThinkBlockOpen_StrictMode_DefersToolParsing()
    {
        // Arrange
        var content = """
            <think>
            Let me think about this... maybe I should call get_weather
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Empty(result.CompletedCalls);
        Assert.True(result.ThinkBlockOpen);
    }

    [Fact]
    public void Parse_ThinkBlockOpen_LenientMode_StillParsesToolCalls()
    {
        // Arrange
        var content = """
            <think>
            reasoning text
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Ankara"}}
            </tool_call>
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: false);

        // Assert
        Assert.Single(result.CompletedCalls);
        Assert.Equal("get_weather", result.CompletedCalls[0].Name);
        Assert.True(result.ThinkBlockOpen);
    }

    [Fact]
    public void Parse_ArgumentsWithNestedTypes_PreservesTypes()
    {
        // Arrange
        var content = """
            <tool_call>
            {"name": "search", "arguments": {"query": "test", "top_k": 5, "include_metadata": true}}
            </tool_call>
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Single(result.CompletedCalls);
        var args = result.CompletedCalls[0].Arguments;
        Assert.Equal("test", args["query"]);
        Assert.Equal(5L, args["top_k"]);
        Assert.Equal(true, args["include_metadata"]);
    }

    [Fact]
    public void Parse_TextBetweenToolCalls_CollectsAllText()
    {
        // Arrange
        var content = """
            First I'll check weather.
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Ankara"}}
            </tool_call>
            Now searching.
            <tool_call>
            {"name": "search", "arguments": {"query": "test"}}
            </tool_call>
            Done.
            """;

        // Act
        var result = NousToolCallParser.Parse(content, strictThink: true);

        // Assert
        Assert.Equal(2, result.CompletedCalls.Count);
        Assert.Contains("First I'll check weather", result.Text);
        Assert.Contains("Now searching", result.Text);
        Assert.Contains("Done", result.Text);
    }
}
