// Copyright (c) Microsoft. All rights reserved.

namespace NousToolCalling;

/// <summary>
/// Configuration options for the <see cref="NousToolCallingChatClient"/> decorator.
/// </summary>
public sealed class NousToolCallingOptions
{
    /// <summary>
    /// Gets or sets a value indicating whether strict think-block mode is enabled.
    /// When <see langword="true"/> (the default), the parser defers all tool-call extraction
    /// while a <c>&lt;think&gt;</c> block is open and <c>&lt;/think&gt;</c> has not yet appeared.
    /// </summary>
    public bool StrictThinkMode { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether think-block content is preserved
    /// in the response message's <c>AdditionalProperties</c> under the key <c>"nous_think"</c>.
    /// </summary>
    public bool PreserveThinkBlocks { get; set; }

    /// <summary>
    /// Gets or sets the prefix used when generating synthetic <c>CallId</c> values
    /// for <see cref="Microsoft.Extensions.AI.FunctionCallContent"/> instances.
    /// Nous-format tool calling has no native call IDs; ordinal-based IDs are generated as
    /// <c>{prefix}_{ordinal}</c>.
    /// </summary>
    public string ToolCallIdPrefix { get; set; } = "nous";
}
