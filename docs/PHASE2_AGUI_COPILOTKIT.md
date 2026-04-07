# Phase 2: AGUI and CopilotKit MCP alignment

This document captures **integration risks and work items** when pairing this stack with **AGUI** (Agent UI protocol/events) and **CopilotKit** (often front-end + MCP-backed tools). It is a spike checklist, not a finished bridge.

## AGUI

- **Event shape:** AGUI clients expect structured streaming events (steps, tool lifecycle, errors). The `NousToolCalling` layer operates on `IChatClient` text/XML; ensure your host translates `AgentResponse` / `AgentResponseUpdate` (or raw `ChatResponseUpdate`) into AGUI-compatible events.
- **Tool visualization:** Nous-style tools still become normal `FunctionCallContent` after parsing; surface the same IDs and JSON arguments your UI already uses for OpenAI-style tool calls.
- **Streaming:** `GetStreamingResponseAsync` through Nous may emit text chunks before tool XML is complete; AGUI layers should tolerate **incomplete** assistant messages until a tool boundary is detected.

## CopilotKit MCP tool

- **Schema parity:** CopilotKit MCP integrations typically advertise tool names and JSON Schema. Ensure MCP server definitions match what you register as `AIFunction` / Nous `<tools>` (argument names and required fields).
- **Server URL vs. embedded stdio:** `McpToolHost` targets **HTTP/SSE** MCP transports supported by the official C# MCP client. If CopilotKit uses a different transport, add an adapter or a small local HTTP gateway.
- **Authentication:** Align headers (Bearer tokens, etc.) between CopilotKit’s MCP configuration and the `McpToolHost` `HttpClient` factory callback.
- **Multi-turn:** CopilotKit may assume OpenAI tool message ordering. Nous reorders tool results into user content with `<tool_response>`; your bridge to CopilotKit should either normalize history or document that only the C# agent path uses Nous XML.

## Suggested next implementation steps

1. Capture one full trace (request/response) from a working CopilotKit + MCP session and compare tool names, arguments, and error payloads to `FunctionInvokingChatClient` + Nous.
2. Prototype an AGUI event mapper on top of `ChatClientAgent.RunStreamingAsync` without changing `NousToolCallingChatClient`.
3. If AGUI requires a specific hosting package (`Microsoft.Agents.AI.AGUI`), add a **separate** sample project that references it and a stub UI—keep the core `NousToolCalling` package free of UI dependencies.
