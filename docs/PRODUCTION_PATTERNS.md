# Production-oriented patterns (Microsoft Agent Framework + NousToolCalling)

This repo demonstrates a minimal production checklist alongside the runnable sample `samples/AgentProdPatterns`.

## Telemetry

- Wrap any `AIAgent` with [`OpenTelemetryAgent`](https://www.nuget.org/packages/Microsoft.Agents.AI/) to emit GenAI semantic convention traces/spans.
- Prefer an OTLP exporter in real deployments instead of the console exporter used in the sample.
- Only enable full message capture (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` or `OpenTelemetryAgent.EnableSensitiveData`) when your pipeline is trusted and compliant; otherwise you risk logging PII.

## Session and persistence

- Use `await agent.CreateSessionAsync()` and pass the same `AgentSession` into subsequent `RunAsync` calls to preserve conversation state handled by the agent (including provider-specific keys in `StateBag`).
- For cross-restart continuity, use `SerializeSessionAsync` / `DeserializeSessionAsync` on the agent and store the JSON in your own secure store.
- Treat deserialized sessions as **untrusted input** until validated (see remarks on `AgentSession` in the framework docs).

## Human-in-the-loop (HITL)

- For dangerous or privileged tools, use patterns such as `ApprovalRequiredAIFunction` (see Microsoft Agent Framework samples under hosted agents and authorization). The thin `NousToolCalling` package does not implement approvals; compose at the `Microsoft.Agents.AI` layer.

## Authorization

- Map HTTP/user identity to `AgentRunOptions` or tool invocations in your host (ASP.NET, Functions, etc.). See the upstream `AspNetAgentAuthorization` sample in the Agent Framework repository for end-to-end patterns.

## MCP and secrets

- Prefer `McpToolHost.ListMcpToolsAsync` / `ListAIToolsAsync` so tools keep the server’s JSON Schema (`McpClientTool` extends `AIFunction`), matching [Microsoft’s MCP tool guidance](https://learn.microsoft.com/agent-framework/user-guide/model-context-protocol/using-mcp-tools).
- When using `McpToolHost`, supply an `HttpClient` per server via the constructor callback so tokens or mTLS live in typed clients rather than in URLs.
