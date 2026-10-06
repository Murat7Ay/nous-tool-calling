// Copyright (c) Murat Ay. Licensed under the MIT License.

namespace NousToolCalling;

/// <summary>
/// The text format the model uses for prompt-based tool calls.
/// </summary>
public enum NousToolCallFormat
{
    /// <summary>
    /// Nous/Hermes JSON format (Qwen, Hermes, DeepSeek, ...):
    /// <c>&lt;tool_call&gt;{"name": ..., "arguments": {...}}&lt;/tool_call&gt;</c>.
    /// </summary>
    Hermes = 0,

    /// <summary>
    /// GLM XML format (GLM-4.5, 4.6, 4.7, GLM-5.x including GLM-5.3-Flash):
    /// <c>&lt;tool_call&gt;name&lt;arg_key&gt;k&lt;/arg_key&gt;&lt;arg_value&gt;v&lt;/arg_value&gt;&lt;/tool_call&gt;</c>.
    /// Argument values are text and are converted using the tool's JSON schema.
    /// </summary>
    Glm = 1,

    /// <summary>
    /// Accepts both formats in the model output, deciding per call: a call body starting with
    /// <c>{</c> is parsed as Hermes JSON, anything else as GLM XML. The tools prompt uses the GLM
    /// format when the model id contains <c>"glm"</c>, otherwise the Hermes format.
    /// </summary>
    Auto = 2,
}
