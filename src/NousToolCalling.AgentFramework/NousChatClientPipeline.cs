// Copyright (c) Microsoft. All rights reserved.

using Microsoft.Extensions.AI;
using NousToolCalling;

namespace NousToolCalling.AgentFramework;

/// <summary>
/// Builds an <see cref="IChatClient"/> stack suitable for Nous/XML tool calling with on-prem models:
/// <see cref="FunctionInvokingChatClient"/> (outermost) then <see cref="NousToolCallingChatClient"/> (inner),
/// then the leaf client (e.g. OpenAI-compatible endpoint).
/// </summary>
public static class NousChatClientPipeline
{
    /// <summary>
    /// Creates the pipeline: FunctionInvocation → NousToolCalling → <paramref name="leafClient"/>.
    /// </summary>
    public static IChatClient Create(IChatClient leafClient, NousToolCallingOptions? nousOptions = null)
    {
        return leafClient
            .AsBuilder()
            .UseFunctionInvocation()
            .UseNousToolCalling(nousOptions ?? new NousToolCallingOptions())
            .Build();
    }
}
