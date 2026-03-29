// Copyright (c) Microsoft. All rights reserved.

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace NousToolCalling;

/// <summary>
/// A delegating chat client that translates between the framework's native tool calling protocol
/// (<see cref="FunctionCallContent"/>/<see cref="FunctionResultContent"/>) and the Nous/XML-based
/// prompt tool calling format used by models like Qwen.
/// </summary>
/// <remarks>
/// <para>
/// When using <see cref="ChatClientBuilder"/>, register <see cref="FunctionInvokingChatClient"/>
/// first (outermost) and this decorator second (innermost, closest to the leaf client):
/// </para>
/// <code>
/// pipeline = leafClient
///     .AsBuilder()
///     .UseFunctionInvocation()       // outermost -- sees FunctionCallContent, invokes tools
///     .UseNousToolCalling(options)    // innermost -- translates XML ↔ FunctionCallContent
///     .Build();
/// </code>
/// <para>
/// On outbound requests, it converts <see cref="ChatOptions.Tools"/> into a <c>&lt;tools&gt;</c> section
/// in the system prompt, converts <see cref="FunctionCallContent"/> in assistant messages to
/// <c>&lt;tool_call&gt;</c> XML, and converts <see cref="FunctionResultContent"/> in tool messages
/// to <c>&lt;tool_response&gt;</c> XML appended to user messages. Tools are stripped from
/// <see cref="ChatOptions"/> to prevent native tool API usage.
/// </para>
/// <para>
/// On inbound responses, it parses <c>&lt;tool_call&gt;</c> blocks from the assistant content and
/// creates <see cref="FunctionCallContent"/> instances that <see cref="FunctionInvokingChatClient"/>
/// can process.
/// </para>
/// </remarks>
public sealed class NousToolCallingChatClient : DelegatingChatClient
{
    private static readonly JsonSerializerOptions s_compactJson = new() { WriteIndented = false };

    private readonly NousToolCallingOptions _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="NousToolCallingChatClient"/> class.
    /// </summary>
    /// <param name="innerClient">The underlying <see cref="IChatClient"/> to delegate to.</param>
    /// <param name="options">Optional configuration for the Nous tool calling behavior.</param>
    public NousToolCallingChatClient(IChatClient innerClient, NousToolCallingOptions? options = null)
        : base(innerClient)
    {
        _options = options ?? new NousToolCallingOptions();
    }

    /// <inheritdoc/>
    public override async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var (transformedMessages, transformedOptions) = TransformRequest(messages, options);

        var response = await base.GetResponseAsync(transformedMessages, transformedOptions, cancellationToken).ConfigureAwait(false);

        return TransformResponse(response);
    }

    /// <inheritdoc/>
    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var (transformedMessages, transformedOptions) = TransformRequest(messages, options);

        var accumulated = new StringBuilder();
        var updates = new List<ChatResponseUpdate>();

        await foreach (var update in base.GetStreamingResponseAsync(transformedMessages, transformedOptions, cancellationToken).ConfigureAwait(false))
        {
            if (update.Text is { Length: > 0 } text)
            {
                accumulated.Append(text);
            }

            updates.Add(update);
            yield return update;
        }

        var fullContent = accumulated.ToString();
        var parseResult = NousToolCallParser.Parse(fullContent, _options.StrictThinkMode);

        if (parseResult.CompletedCalls.Count > 0)
        {
            var toolCallContents = new List<AIContent>();
            foreach (var call in parseResult.CompletedCalls)
            {
                var callId = $"{_options.ToolCallIdPrefix}_{call.Ordinal}";
                toolCallContents.Add(new FunctionCallContent(callId, call.Name, call.Arguments));
            }

            var lastUpdate = updates.Count > 0 ? updates[^1] : null;

            yield return new ChatResponseUpdate
            {
                Role = ChatRole.Assistant,
                Contents = toolCallContents,
                ResponseId = lastUpdate?.ResponseId,
                MessageId = lastUpdate?.MessageId,
                ConversationId = lastUpdate?.ConversationId,
            };
        }
    }

    private (List<ChatMessage> Messages, ChatOptions? Options) TransformRequest(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options)
    {
        var messageList = messages.ToList();
        var tools = options?.Tools;

        string? toolsSection = null;
        if (tools is { Count: > 0 })
        {
            toolsSection = NousToolPromptBuilder.BuildToolsSection(tools);
        }

        var transformed = new List<ChatMessage>(messageList.Count);

        for (int i = 0; i < messageList.Count; i++)
        {
            var msg = messageList[i];

            if (msg.Role == ChatRole.System && toolsSection is { Length: > 0 } && i == FindFirstSystemIndex(messageList))
            {
                var existingText = GetTextContent(msg);
                var mergedContent = NousToolPromptBuilder.MergeIntoSystem(existingText, toolsSection);
                transformed.Add(new ChatMessage(ChatRole.System, mergedContent));
                toolsSection = null;
            }
            else if (msg.Role == ChatRole.Assistant && HasFunctionCallContent(msg))
            {
                transformed.Add(ConvertAssistantMessage(msg));
            }
            else if (msg.Role == ChatRole.Tool)
            {
                AppendToolResponseMessages(transformed, msg);
            }
            else
            {
                transformed.Add(msg);
            }
        }

        if (toolsSection is { Length: > 0 })
        {
            transformed.Insert(0, new ChatMessage(ChatRole.System, toolsSection));
        }

        ChatOptions? transformedOptions = null;
        if (options is not null)
        {
            transformedOptions = options.Clone();
            transformedOptions.Tools = null;
        }

        return (transformed, transformedOptions);
    }

    private ChatResponse TransformResponse(ChatResponse response)
    {
        foreach (var message in response.Messages)
        {
            if (message.Role != ChatRole.Assistant)
            {
                continue;
            }

            var textContent = GetTextContent(message);
            if (string.IsNullOrEmpty(textContent))
            {
                continue;
            }

            var parseResult = NousToolCallParser.Parse(textContent, _options.StrictThinkMode);

            if (parseResult.CompletedCalls.Count == 0)
            {
                MaybeStoreThinkContent(message, parseResult);
                continue;
            }

            var newContents = new List<AIContent>();

            if (parseResult.Text.Length > 0)
            {
                newContents.Add(new TextContent(parseResult.Text));
            }

            foreach (var call in parseResult.CompletedCalls)
            {
                var callId = $"{_options.ToolCallIdPrefix}_{call.Ordinal}";
                newContents.Add(new FunctionCallContent(callId, call.Name, call.Arguments));
            }

            message.Contents.Clear();
            foreach (var content in newContents)
            {
                message.Contents.Add(content);
            }

            MaybeStoreThinkContent(message, parseResult);

            response.FinishReason = ChatFinishReason.ToolCalls;
        }

        return response;
    }

    private static ChatMessage ConvertAssistantMessage(ChatMessage original)
    {
        var sb = new StringBuilder();

        foreach (var content in original.Contents)
        {
            if (content is TextContent tc && !string.IsNullOrEmpty(tc.Text))
            {
                sb.Append(tc.Text);
            }
            else if (content is FunctionCallContent fcc)
            {
                if (sb.Length > 0 && sb[^1] != '\n')
                {
                    sb.Append('\n');
                }

                sb.Append(NousToolPromptBuilder.FormatToolCall(fcc.Name, fcc.Arguments));
            }
        }

        return new ChatMessage(ChatRole.Assistant, sb.ToString())
        {
            AuthorName = original.AuthorName,
            MessageId = original.MessageId,
        };
    }

    private static void AppendToolResponseMessages(List<ChatMessage> target, ChatMessage toolMessage)
    {
        var sb = new StringBuilder();
        foreach (var content in toolMessage.Contents)
        {
            if (content is FunctionResultContent frc)
            {
                var resultText = frc.Result is string s ? s : JsonSerializer.Serialize(frc.Result, s_compactJson);
                if (sb.Length > 0)
                {
                    sb.Append('\n');
                }

                sb.Append(NousToolPromptBuilder.FormatToolResponse(resultText));
            }
        }

        if (sb.Length == 0)
        {
            return;
        }

        var wrapped = sb.ToString();

        if (target.Count > 0 && target[^1].Role == ChatRole.User)
        {
            var last = target[^1];
            var existingText = GetTextContent(last);
            var separator = string.IsNullOrWhiteSpace(existingText) ? "" : "\n";
            target[^1] = new ChatMessage(ChatRole.User, $"{existingText}{separator}{wrapped}")
            {
                AuthorName = last.AuthorName,
                MessageId = last.MessageId,
            };
        }
        else
        {
            target.Add(new ChatMessage(ChatRole.User, wrapped));
        }
    }

    private static string GetTextContent(ChatMessage message)
    {
        var sb = new StringBuilder();
        foreach (var content in message.Contents)
        {
            if (content is TextContent tc)
            {
                sb.Append(tc.Text);
            }
        }

        return sb.ToString();
    }

    private static bool HasFunctionCallContent(ChatMessage message)
    {
        foreach (var content in message.Contents)
        {
            if (content is FunctionCallContent)
            {
                return true;
            }
        }

        return false;
    }

    private static int FindFirstSystemIndex(List<ChatMessage> messages)
    {
        for (int i = 0; i < messages.Count; i++)
        {
            if (messages[i].Role == ChatRole.System)
            {
                return i;
            }
        }

        return -1;
    }

    private void MaybeStoreThinkContent(ChatMessage message, NousParseResult parseResult)
    {
        if (_options.PreserveThinkBlocks && parseResult.ThinkContent is { Length: > 0 })
        {
            message.AdditionalProperties ??= new();
            message.AdditionalProperties["nous_think"] = parseResult.ThinkContent;
        }
    }
}
