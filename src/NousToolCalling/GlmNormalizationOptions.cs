// Copyright (c) Murat Ay. Licensed under the MIT License.

namespace NousToolCalling;

/// <summary>
/// Settings for <see cref="GlmNormalizationHandler"/>. Every fix is on by default except
/// <see cref="EnableThinking"/>, which is left out of requests unless set.
/// </summary>
public sealed class GlmNormalizationOptions
{
    /// <summary>
    /// Gets or sets a value indicating whether <c>reasoning</c> / <c>reasoning_content</c> are removed from the
    /// messages sent upstream, so the history carries only content and tool calls. Default <see langword="true"/>.
    /// </summary>
    public bool StripReasoningFromHistory { get; set; } = true;

    /// <summary>
    /// Gets or sets the value sent as <c>chat_template_kwargs.enable_thinking</c>. When <see langword="null"/> (the default)
    /// the key is removed from every request: on servers where thinking cannot be turned off (GLM-5.3), sending
    /// <c>false</c> makes the reasoning leak into the content.
    /// </summary>
    public bool? EnableThinking { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether <c>&lt;think&gt;…&lt;/think&gt;</c> (or text before a lone <c>&lt;/think&gt;</c>)
    /// is moved out of non-streaming content into the reasoning. Default <see langword="true"/>.
    /// </summary>
    public bool StripThinkTags { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether the reasoning is used as the content when a choice ends with empty
    /// content and no tool calls. Default <see langword="true"/>.
    /// </summary>
    public bool ReasoningAsContentFallback { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether <c>reasoning</c> is also copied to <c>reasoning_content</c> (the field
    /// Microsoft.Extensions.AI.OpenAI reads into <c>TextReasoningContent</c>). Default <see langword="true"/>.
    /// </summary>
    public bool MirrorReasoningContent { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether literal <c>\uXXXX</c> text in content and reasoning is decoded
    /// (surrogate pairs joined; code points below 0x20 left as written). Default <see langword="true"/>.
    /// </summary>
    public bool DecodeUnicodeEscapes { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether stream chunks with an empty <c>choices</c> array (vLLM's final
    /// usage-only chunk) are dropped. Default <see langword="true"/>. Set to <see langword="false"/> to keep token usage
    /// when your client handles such chunks.
    /// </summary>
    public bool DropEmptyChoicesChunks { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether upstream failures are made readable: a stream that breaks mid-answer
    /// is closed cleanly, a connection failure becomes an HTTP 502 with an OpenAI-style error body, and bare vLLM
    /// error bodies (<c>{"object":"error",…}</c>) are wrapped in <c>{"error": …}</c>. Default <see langword="true"/>.
    /// </summary>
    public bool HandleUpstreamErrors { get; set; } = true;
}
