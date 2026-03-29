// Copyright (c) Microsoft. All rights reserved.

using NousToolCalling;

namespace Microsoft.Extensions.AI;

/// <summary>
/// Provides extension methods for adding <see cref="NousToolCallingChatClient"/> to a chat client pipeline.
/// </summary>
public static class NousToolCallingChatClientExtensions
{
    /// <summary>
    /// Adds a <see cref="NousToolCallingChatClient"/> to the chat client pipeline that translates
    /// between the framework's native tool calling protocol and the Nous/XML-based prompt tool calling
    /// format used by models like Qwen.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Pipeline order matters.</b> In <see cref="ChatClientBuilder"/>, the first <c>Use</c> call
    /// becomes the outermost layer. <see cref="FunctionInvokingChatClient"/> must be outermost so it
    /// can see the <see cref="Microsoft.Extensions.AI.FunctionCallContent"/> instances that this
    /// decorator produces. Register it <b>before</b> this method:
    /// </para>
    /// <code>
    /// pipeline = leafClient
    ///     .AsBuilder()
    ///     .UseFunctionInvocation()       // outermost
    ///     .UseNousToolCalling(options)    // innermost
    ///     .Build();
    /// </code>
    /// </remarks>
    /// <param name="builder">The <see cref="ChatClientBuilder"/> to add the decorator to.</param>
    /// <param name="options">Optional configuration for the Nous tool calling behavior.</param>
    /// <returns>The <paramref name="builder"/> for chaining.</returns>
    public static ChatClientBuilder UseNousToolCalling(this ChatClientBuilder builder, NousToolCallingOptions? options = null)
    {
        return builder.Use(innerClient => new NousToolCallingChatClient(innerClient, options));
    }
}
