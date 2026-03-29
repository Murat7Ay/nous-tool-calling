// Copyright (c) Microsoft. All rights reserved.

using NousToolCalling;
using Microsoft.Extensions.AI;
using Xunit;

namespace NousToolCalling.UnitTests;

public class NousToolPromptBuilderTests
{
    [Fact]
    public void BuildToolsSection_WithAIFunction_GeneratesCorrectFormat()
    {
        // Arrange
        var tool = AIFunctionFactory.Create(
            (string city) => $"Weather in {city}: sunny",
            name: "get_weather",
            description: "Gets the weather for a city");

        // Act
        var result = NousToolPromptBuilder.BuildToolsSection([tool]);

        // Assert
        Assert.Contains("<tools>", result);
        Assert.Contains("</tools>", result);
        Assert.Contains("get_weather", result);
        Assert.Contains("Gets the weather for a city", result);
        Assert.Contains("<tool_call>", result);
    }

    [Fact]
    public void BuildToolsSection_WithMultipleTools_IncludesAll()
    {
        // Arrange
        var tool1 = AIFunctionFactory.Create(
            (string city) => $"Weather in {city}",
            name: "get_weather",
            description: "Gets weather");

        var tool2 = AIFunctionFactory.Create(
            (string query) => $"Results for {query}",
            name: "search_kb",
            description: "Searches knowledge base");

        // Act
        var result = NousToolPromptBuilder.BuildToolsSection([tool1, tool2]);

        // Assert
        Assert.Contains("get_weather", result);
        Assert.Contains("search_kb", result);
    }

    [Fact]
    public void BuildToolsSection_WithEmptyList_ReturnsEmpty()
    {
        // Act
        var result = NousToolPromptBuilder.BuildToolsSection([]);

        // Assert
        Assert.Equal(string.Empty, result);
    }

    [Fact]
    public void MergeIntoSystem_WithExistingInstructions_CombinesBoth()
    {
        // Arrange
        var existing = "You are a helpful assistant.";
        var toolsSection = "# Tools\n<tools>...</tools>";

        // Act
        var result = NousToolPromptBuilder.MergeIntoSystem(existing, toolsSection);

        // Assert
        Assert.StartsWith("You are a helpful assistant.", result);
        Assert.Contains("# Tools", result);
    }

    [Fact]
    public void MergeIntoSystem_WithNullExisting_ReturnsToolsOnly()
    {
        // Arrange
        var toolsSection = "# Tools\n<tools>...</tools>";

        // Act
        var result = NousToolPromptBuilder.MergeIntoSystem(null, toolsSection);

        // Assert
        Assert.Equal("# Tools\n<tools>...</tools>", result);
    }

    [Fact]
    public void FormatToolCall_ProducesCorrectXml()
    {
        // Arrange
        var args = new Dictionary<string, object?> { ["city"] = "Ankara" };

        // Act
        var result = NousToolPromptBuilder.FormatToolCall("get_weather", args);

        // Assert
        Assert.StartsWith("<tool_call>", result);
        Assert.EndsWith("</tool_call>", result);
        Assert.Contains("get_weather", result);
        Assert.Contains("Ankara", result);
    }

    [Fact]
    public void FormatToolResponse_WrapsCorrectly()
    {
        // Arrange
        var resultText = """{"city":"Ankara","summary":"Sunny, 22C"}""";

        // Act
        var result = NousToolPromptBuilder.FormatToolResponse(resultText);

        // Assert
        Assert.StartsWith("<tool_response>", result);
        Assert.EndsWith("</tool_response>", result);
        Assert.Contains("Ankara", result);
    }
}
