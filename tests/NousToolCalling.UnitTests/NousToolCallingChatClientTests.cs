// Copyright (c) Microsoft. All rights reserved.

using System.Text.Json;
using NousToolCalling;
using Microsoft.Extensions.AI;
using Xunit;

namespace NousToolCalling.UnitTests;

public class NousToolCallingChatClientTests
{
    [Fact]
    public async Task GetResponseAsync_InjectsToolsIntoSystemPrompt()
    {
        // Arrange
        string? capturedSystemPrompt = null;

        var innerClient = new TestChatClient((messages, options) =>
        {
            capturedSystemPrompt = messages
                .FirstOrDefault(m => m.Role == ChatRole.System)
                ?.Text;

            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, "Hello")));
        });

        var tool = AIFunctionFactory.Create(
            (string city) => $"Weather in {city}",
            name: "get_weather",
            description: "Gets weather");

        var client = new NousToolCallingChatClient(innerClient);

        var chatOptions = new ChatOptions { Tools = [tool] };
        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, "You are helpful."),
            new(ChatRole.User, "What's the weather?"),
        };

        // Act
        await client.GetResponseAsync(messages, chatOptions);

        // Assert
        Assert.NotNull(capturedSystemPrompt);
        Assert.Contains("<tools>", capturedSystemPrompt);
        Assert.Contains("get_weather", capturedSystemPrompt);
        Assert.Contains("You are helpful.", capturedSystemPrompt);
    }

    [Fact]
    public async Task GetResponseAsync_StripsToolsFromOptions()
    {
        // Arrange
        ChatOptions? capturedOptions = null;

        var innerClient = new TestChatClient((messages, options) =>
        {
            capturedOptions = options;
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, "Hi")));
        });

        var tool = AIFunctionFactory.Create(
            (string city) => $"Weather in {city}",
            name: "get_weather",
            description: "Gets weather");

        var client = new NousToolCallingChatClient(innerClient);
        var chatOptions = new ChatOptions { Tools = [tool] };

        // Act
        await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")], chatOptions);

        // Assert
        Assert.Null(capturedOptions?.Tools);
    }

    [Fact]
    public async Task GetResponseAsync_ParsesToolCallFromResponse()
    {
        // Arrange
        var responseContent = """
            Let me check.
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Ankara"}}
            </tool_call>
            """;

        var innerClient = new TestChatClient((_, _) =>
            Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, responseContent))));

        var client = new NousToolCallingChatClient(innerClient);

        // Act
        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Weather?")]);

        // Assert
        var message = response.Messages[0];
        Assert.Equal(ChatRole.Assistant, message.Role);

        var textContents = message.Contents.OfType<TextContent>().ToList();
        var functionCalls = message.Contents.OfType<FunctionCallContent>().ToList();

        Assert.Single(functionCalls);
        Assert.Equal("get_weather", functionCalls[0].Name);
        Assert.Equal("Ankara", functionCalls[0].Arguments?["city"]?.ToString());

        Assert.Single(textContents);
        Assert.Contains("Let me check", textContents[0].Text);
    }

    [Fact]
    public async Task GetResponseAsync_ConvertsFunctionCallContentToNousFormat()
    {
        // Arrange
        List<ChatMessage>? capturedMessages = null;

        var innerClient = new TestChatClient((messages, _) =>
        {
            capturedMessages = messages.ToList();
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, "Ankara is sunny.")));
        });

        var client = new NousToolCallingChatClient(innerClient);

        var messages = new List<ChatMessage>
        {
            new(ChatRole.System, "You are helpful."),
            new(ChatRole.User, "Weather in Ankara?"),
            new(ChatRole.Assistant, [
                new TextContent("Let me check."),
                new FunctionCallContent("nous_1", "get_weather", new Dictionary<string, object?> { ["city"] = "Ankara" }),
            ]),
            new(ChatRole.Tool, [
                new FunctionResultContent("nous_1", """{"city":"Ankara","temp":"22C"}"""),
            ]),
        };

        // Act
        await client.GetResponseAsync(messages);

        // Assert
        Assert.NotNull(capturedMessages);

        var assistantMsg = capturedMessages.First(m => m.Role == ChatRole.Assistant);
        Assert.Contains("<tool_call>", assistantMsg.Text);
        Assert.Contains("get_weather", assistantMsg.Text);
        Assert.Contains("Let me check.", assistantMsg.Text);

        var hasToolResponse = capturedMessages.Any(m =>
            m.Role == ChatRole.User && m.Text is not null && m.Text.Contains("<tool_response>"));
        Assert.True(hasToolResponse);

        Assert.DoesNotContain(capturedMessages, m => m.Role == ChatRole.Tool);
    }

    [Fact]
    public async Task GetResponseAsync_NoSystemMessage_CreatesOneWithTools()
    {
        // Arrange
        List<ChatMessage>? capturedMessages = null;

        var innerClient = new TestChatClient((messages, _) =>
        {
            capturedMessages = messages.ToList();
            return Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, "Hi")));
        });

        var tool = AIFunctionFactory.Create(
            (string city) => $"Weather in {city}",
            name: "get_weather",
            description: "Gets weather");

        var client = new NousToolCallingChatClient(innerClient);
        var chatOptions = new ChatOptions { Tools = [tool] };

        // Act
        await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Hi")], chatOptions);

        // Assert
        Assert.NotNull(capturedMessages);
        var systemMsg = capturedMessages.FirstOrDefault(m => m.Role == ChatRole.System);
        Assert.NotNull(systemMsg);
        Assert.Contains("<tools>", systemMsg.Text);
    }

    [Fact]
    public async Task GetResponseAsync_ParallelToolCalls_ExtractsBoth()
    {
        // Arrange
        var responseContent = """
            I'll check both.
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Ankara"}}
            </tool_call>
            <tool_call>
            {"name": "search_kb", "arguments": {"query": "leave policy"}}
            </tool_call>
            """;

        var innerClient = new TestChatClient((_, _) =>
            Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, responseContent))));

        var client = new NousToolCallingChatClient(innerClient);

        // Act
        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Test")]);

        // Assert
        var functionCalls = response.Messages[0].Contents.OfType<FunctionCallContent>().ToList();
        Assert.Equal(2, functionCalls.Count);
        Assert.Equal("get_weather", functionCalls[0].Name);
        Assert.Equal("search_kb", functionCalls[1].Name);
    }

    [Fact]
    public async Task GetResponseAsync_WithThinkBlock_PreservesInMetadata()
    {
        // Arrange
        var responseContent = """
            <think>
            I should look up the weather.
            </think>
            <tool_call>
            {"name": "get_weather", "arguments": {"city": "Ankara"}}
            </tool_call>
            """;

        var innerClient = new TestChatClient((_, _) =>
            Task.FromResult(new ChatResponse(new ChatMessage(ChatRole.Assistant, responseContent))));

        var client = new NousToolCallingChatClient(innerClient, new NousToolCallingOptions { PreserveThinkBlocks = true });

        // Act
        var response = await client.GetResponseAsync([new ChatMessage(ChatRole.User, "Test")]);

        // Assert
        var message = response.Messages[0];
        Assert.NotNull(message.AdditionalProperties);
        Assert.True(message.AdditionalProperties.ContainsKey("nous_think"));
        Assert.Contains("look up the weather", message.AdditionalProperties["nous_think"]?.ToString());
    }

    /// <summary>
    /// Simple test IChatClient that delegates to a callback.
    /// </summary>
    private sealed class TestChatClient : IChatClient
    {
        private readonly Func<IEnumerable<ChatMessage>, ChatOptions?, Task<ChatResponse>> _handler;

        public TestChatClient(Func<IEnumerable<ChatMessage>, ChatOptions?, Task<ChatResponse>> handler)
        {
            _handler = handler;
        }

        public Task<ChatResponse> GetResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
            => _handler(messages, options);

        public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
            IEnumerable<ChatMessage> messages,
            ChatOptions? options = null,
            CancellationToken cancellationToken = default)
            => throw new NotImplementedException();

        public object? GetService(Type serviceType, object? serviceKey = null) => null;

        public void Dispose() { }
    }
}
