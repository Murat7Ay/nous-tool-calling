// Copyright (c) Microsoft. All rights reserved.

using System.Globalization;
using System.Net.Http;
using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;

namespace NousToolCalling.AgentFramework;

/// <summary>
/// Connects to remote MCP servers over HTTP (streamable HTTP / SSE via <see cref="HttpTransportMode"/>)
/// and surfaces server tools the way the official MCP C# SDK and Microsoft docs intend:
/// <see cref="McpClient.ListToolsAsync"/> → <see cref="McpClientTool"/> (inherits <see cref="AIFunction"/> with server <c>inputSchema</c>).
/// </summary>
/// <remarks>
/// See: https://learn.microsoft.com/dotnet/ai/quickstarts/build-mcp-client and
/// https://learn.microsoft.com/agent-framework/user-guide/model-context-protocol/using-mcp-tools —
/// tools from <see cref="ListMcpToolsAsync"/> are passed directly to <see cref="ChatOptions.Tools"/> / agent constructors as <see cref="AITool"/>.
/// </remarks>
public sealed class McpToolHost : IAsyncDisposable
{
    private readonly Func<string, CancellationToken, Task<HttpClient?>>? _httpClientProvider;
    private readonly Dictionary<string, McpClient> _clients = new();
    private readonly Dictionary<string, HttpClient> _ownedHttpClients = new();
    private readonly SemaphoreSlim _clientLock = new(1, 1);

    /// <param name="httpClientProvider">
    /// Optional per-server <see cref="HttpClient"/> factory for auth or proxies.
    /// Return <see langword="null"/> to use a default client for that URL.
    /// </param>
    public McpToolHost(Func<string, CancellationToken, Task<HttpClient?>>? httpClientProvider = null)
    {
        _httpClientProvider = httpClientProvider;
    }

    /// <summary>
    /// Lists tools from the MCP server. Each <see cref="McpClientTool"/> is already wired to invoke the server with correct schemas.
    /// </summary>
    public async Task<IReadOnlyList<McpClientTool>> ListMcpToolsAsync(
        string serverUrl,
        string? serverLabel = null,
        IDictionary<string, string>? headers = null,
        CancellationToken cancellationToken = default)
    {
        McpClient client = await GetOrCreateClientAsync(serverUrl, serverLabel, headers, cancellationToken).ConfigureAwait(false);
        IList<McpClientTool> tools = await client.ListToolsAsync(cancellationToken: cancellationToken).ConfigureAwait(false);
        return tools is List<McpClientTool> list ? list : tools.ToList();
    }

    /// <summary>
    /// Convenience: same as <see cref="ListMcpToolsAsync"/> but typed as <see cref="AITool"/> for agent/chat options.
    /// </summary>
    public async Task<IReadOnlyList<AITool>> ListAIToolsAsync(
        string serverUrl,
        string? serverLabel = null,
        IDictionary<string, string>? headers = null,
        CancellationToken cancellationToken = default)
    {
        IReadOnlyList<McpClientTool> tools = await ListMcpToolsAsync(serverUrl, serverLabel, headers, cancellationToken)
            .ConfigureAwait(false);
        return tools.Cast<AITool>().ToList();
    }

    private async Task<McpClient> GetOrCreateClientAsync(
        string serverUrl,
        string? serverLabel,
        IDictionary<string, string>? headers,
        CancellationToken cancellationToken)
    {
        string normalizedUrl = serverUrl.Trim().ToUpperInvariant();
        string clientCacheKey = $"{normalizedUrl}|{ComputeHeadersHash(headers)}";

        await _clientLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_clients.TryGetValue(clientCacheKey, out McpClient? existing))
            {
                return existing;
            }

            McpClient created = await CreateClientAsync(serverUrl, serverLabel, headers, normalizedUrl, cancellationToken)
                .ConfigureAwait(false);
            _clients[clientCacheKey] = created;
            return created;
        }
        finally
        {
            _clientLock.Release();
        }
    }

    private async Task<McpClient> CreateClientAsync(
        string serverUrl,
        string? serverLabel,
        IDictionary<string, string>? headers,
        string httpClientCacheKey,
        CancellationToken cancellationToken)
    {
        HttpClient? httpClient = null;
        if (_httpClientProvider is not null)
        {
            httpClient = await _httpClientProvider(serverUrl, cancellationToken).ConfigureAwait(false);
        }

        if (httpClient is null && !_ownedHttpClients.TryGetValue(httpClientCacheKey, out httpClient))
        {
            httpClient = new HttpClient();
            _ownedHttpClients[httpClientCacheKey] = httpClient;
        }

        HttpClientTransportOptions transportOptions = new()
        {
            Endpoint = new Uri(serverUrl),
            Name = serverLabel ?? "McpToolHost",
            AdditionalHeaders = headers,
            TransportMode = HttpTransportMode.AutoDetect,
        };

        HttpClientTransport transport = new(transportOptions, httpClient);
        return await McpClient.CreateAsync(transport, cancellationToken: cancellationToken).ConfigureAwait(false);
    }

    private static string ComputeHeadersHash(IDictionary<string, string>? headers)
    {
        if (headers is null || headers.Count == 0)
        {
            return string.Empty;
        }

        SortedDictionary<string, string> sorted = new(
            headers.ToDictionary(h => h.Key.ToUpperInvariant(), h => h.Value.ToUpperInvariant()));
        int hashCode = 17;
        foreach (KeyValuePair<string, string> kvp in sorted)
        {
            hashCode = (hashCode * 31) + StringComparer.OrdinalIgnoreCase.GetHashCode(kvp.Key);
            hashCode = (hashCode * 31) + StringComparer.OrdinalIgnoreCase.GetHashCode(kvp.Value);
        }

        return hashCode.ToString(CultureInfo.InvariantCulture);
    }

    /// <inheritdoc />
    public async ValueTask DisposeAsync()
    {
        await _clientLock.WaitAsync().ConfigureAwait(false);
        try
        {
            foreach (McpClient client in _clients.Values)
            {
                await client.DisposeAsync().ConfigureAwait(false);
            }

            _clients.Clear();

            foreach (HttpClient http in _ownedHttpClients.Values)
            {
                http.Dispose();
            }

            _ownedHttpClients.Clear();
        }
        finally
        {
            _clientLock.Release();
        }

        _clientLock.Dispose();
    }
}
