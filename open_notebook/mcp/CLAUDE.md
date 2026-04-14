# MCP Module

Model Context Protocol (MCP) client integration enabling the chat AI to call tools from external MCP servers (Composio, custom servers, etc.).

## Purpose

Provides a complete MCP client stack: configuration management (JSON file persistence with encrypted headers), a JSON-RPC client supporting both Streamable HTTP and SSE transports, and a LangChain bridge that converts MCP tool definitions into `StructuredTool` objects compatible with the chat graph.

## Architecture Overview

```
chat.py → _get_mcp_tools()
              ↓
langchain_bridge.py → get_mcp_langchain_tools()
              ↓
    ┌─────────┴──────────┐
    │  client.py          │ ← MCPClient (JSON-RPC over HTTP/SSE)
    │  config.py          │ ← MCPConfigManager (JSON file + encryption)
    └─────────────────────┘
              ↓
    External MCP Servers (Composio, Garza MCP, etc.)
```

**Data flow**: Chat graph calls `_get_mcp_tools()` → bridge loads enabled server configs → creates/reuses cached `MCPClient` per server → discovers tools via `tools/list` → converts to LangChain `StructuredTool` with Pydantic arg schemas → tools bound to chat model for tool calling.

## Component Catalog

### `__init__.py`
Public API: exports `MCPClient`, `MCPConfigManager`, `get_mcp_langchain_tools`.

### `config.py` — Server Configuration Management

**MCPServerConfig** (Pydantic BaseModel):
- `id`: Unique server identifier (e.g., `mcp_8df240661df9`)
- `name`: Human-readable name
- `url`: MCP server endpoint URL
- `headers`: HTTP headers (e.g., Authorization) — encrypted at rest via Fernet
- `enabled`: Whether server is active
- `description`: Optional description

**MCPConfigManager**:
- Persists server configs to `{DATA_FOLDER}/mcp_servers.json`
- Encrypts header values before writing, decrypts after reading
- CRUD operations: `list_servers()`, `get_server()`, `add_server()`, `update_server()`, `delete_server()`
- `get_enabled_servers()`: Returns only active servers
- Global singleton: `mcp_config_manager`

**Header Encryption**:
- `_encrypt_headers()`: Encrypts each header value using Fernet (via `open_notebook.utils.encryption`)
- `_decrypt_headers()`: Decrypts on read
- Uses `looks_like_fernet_token()` to avoid double-encrypting
- Falls back to plaintext if encryption key not configured (dev mode)

### `client.py` — MCP JSON-RPC Client

**MCPClient**: Async client communicating with MCP servers.

**Transport Detection** (automatic):
1. First tries **Streamable HTTP** (POST JSON-RPC to URL)
2. Falls back to **SSE** (GET `/sse` → persistent stream, POST to `/messages/?session_id=...`)

**Key Methods**:
- `initialize()`: MCP handshake — sends `initialize` + `notifications/initialized`
- `list_tools()`: Discovers available tools via `tools/list`
- `call_tool(name, arguments)`: Executes a tool via `tools/call`
- `test_connection()`: Resets state, initializes, lists tools, returns status

**SSE Session Management**:
- `_ensure_sse_session()`: Opens persistent SSE stream, reads endpoint event
- `_sse_request()`: Sends request + reads matching response from stream
- `_close_sse()`: Cleans up SSE client, response, and buffer
- Session state: `_sse_client`, `_sse_response`, `_sse_message_url`, `_sse_buffer`

### `langchain_bridge.py` — MCP-to-LangChain Bridge

**Core Function**: `get_mcp_langchain_tools()` — main entry point for the chat graph.

**Dedicated Event Loop**:
- MCP operations run on a single background thread (`mcp-event-loop`) so SSE connections persist across tool calls
- `_get_mcp_loop()`: Creates/returns the dedicated event loop
- `_run_async(coro)`: Dispatches coroutines to this loop via `run_coroutine_threadsafe()`

**Client Caching**:
- `_client_cache`: Dict mapping server ID → `MCPClient` instance
- Clients reuse SSE sessions across multiple tool calls

**Schema Sanitization** (Gemini compatibility):
- `_json_schema_to_pydantic_field()`: Converts JSON Schema properties to Pydantic fields
- **Avoids `Optional[T]`**: Uses concrete defaults instead (empty string, 0, False) because Pydantic v2's `anyOf` unions break Gemini
- `_build_args_model()`: Creates Pydantic model from MCP tool's `inputSchema`
- **Empty properties handling**: Adds dummy `placeholder` field for tools with no inputs (Gemini rejects empty `properties`)

**Placeholder Filtering**:
- `_make_tool_func()`: The callable wraps `client.call_tool()` and strips the `placeholder` param before sending to the MCP server

**Tool Naming**: `{server_config.id}__{tool_name}` — double underscore separates server ID from tool name.

## Important Patterns

### Dedicated Event Loop for SSE Persistence
Unlike workspace tools (which create a new event loop per call), MCP tools use a single persistent background thread. This is critical for SSE transport where the server maintains session state:

```python
_mcp_loop = asyncio.new_event_loop()
_mcp_thread = threading.Thread(target=_mcp_loop.run_forever, daemon=True)
_mcp_thread.start()
```

### Gemini Schema Sanitization
Gemini's function-calling API has strict JSON Schema requirements:
1. No `anyOf` unions → avoid `Optional[T]`
2. No empty `properties` → add dummy placeholder
3. Arrays need `items.type` → infer from schema or default to `List[str]`

### Error Recovery
Tool errors return error strings instead of raising, keeping the chat graph alive:

```python
def tool_func(**kwargs):
    try:
        result = _run_async(client.call_tool(tool_name, kwargs))
        if isinstance(result, dict) and "error" in result:
            return f"Error: {result['error']}"
        return result
    except Exception as e:
        return f"Error executing tool: {e}"
```

## Key Dependencies

- `httpx`: Async HTTP client for JSON-RPC and SSE transports
- `langchain_core.tools`: `StructuredTool` for LangChain compatibility
- `pydantic`: `BaseModel`, `Field`, `create_model` for dynamic arg schemas
- `open_notebook.mcp.config`: Server configuration and encryption
- `open_notebook.utils.encryption`: Fernet encrypt/decrypt for header values
- `open_notebook.config`: `DATA_FOLDER` for config file location
- `loguru`: Logging

## Quirks & Gotchas

- **SSE session persistence**: The dedicated event loop thread is a daemon thread — it dies when the main process exits. No graceful shutdown by default.
- **180-second timeout**: `_run_async()` has a hard 180s timeout for MCP tool calls. Long-running tools may hit this.
- **Client cache never auto-expires**: `_client_cache` grows unbounded. Call `clear_client_cache()` to reset.
- **SSE buffer parsing**: `_sse_read_response()` matches responses by `id` field. If the server sends events out of order, earlier responses may be lost.
- **Transport detection is per-initialize**: Once detected, transport type is fixed for the client's lifetime. Server changes require cache clear.
- **JSON-RPC notifications have no response**: `_sse_request()` returns `{}` for notifications (no `id` field).
- **Config file locking**: `MCPConfigManager` uses no file locking. Concurrent writes from multiple processes could corrupt the JSON file.
- **Tool count can be large**: Composio returns 500+ tools; Garza MCP returns 178. The model may struggle with too many tools. Consider filtering.
- **Placeholder param leaks if not filtered**: `_make_tool_func()` pops `placeholder` before calling MCP server. If a tool genuinely has a `placeholder` param, it would be stripped.

## How to Extend

1. **Add transport type**: Extend `MCPClient.initialize()` with new transport detection logic; add transport-specific `_post_jsonrpc()` handler
2. **Add tool filtering**: Modify `get_mcp_langchain_tools()` to accept filter criteria (by name pattern, description keyword, etc.)
3. **Add connection pooling**: Replace `_client_cache` dict with an LRU cache that auto-evicts stale clients
4. **Add retry logic**: Wrap `_run_async()` with retry/backoff for transient failures
5. **Add server health monitoring**: Periodic `test_connection()` calls to detect stale SSE sessions

## API Router Integration

The MCP module is exposed via `api/routers/mcp.py`:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/mcp/servers` | List all servers (headers masked via `_mask_headers()`) |
| POST | `/mcp/servers` | Add a new server |
| PUT | `/mcp/servers/{id}` | Update server (skips headers if not provided) |
| DELETE | `/mcp/servers/{id}` | Delete server |
| POST | `/mcp/servers/{id}/test` | Test connection |
| GET | `/mcp/servers/{id}/tools` | List server tools |

**Security**: API responses mask header values as `****[last4chars]` to prevent credential exposure. Edit dialog tracks `headersModified` state to prevent masked values from overwriting real credentials on save.
