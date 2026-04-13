"""
MCP client supporting both Streamable HTTP and SSE transports.
Connects to MCP servers, performs initialization handshake, and discovers tools.

Transport detection:
- First tries Streamable HTTP (POST to URL with JSON-RPC)
- If that returns 404/405, falls back to SSE (GET /sse → POST /messages/?session_id=...)

SSE transport note:
  In SSE, POST requests return 202 Accepted immediately. The actual JSON-RPC
  response arrives via the SSE event stream. We keep the stream open and
  correlate responses by JSON-RPC id.
"""

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from open_notebook.mcp.config import MCPServerConfig


class MCPClient:
    """Client for communicating with MCP servers via Streamable HTTP or SSE."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session_id: Optional[str] = None
        self._initialized = False
        self._transport: Optional[str] = None  # "streamable_http" or "sse"
        self._sse_message_url: Optional[str] = None
        # SSE stream state
        self._sse_responses: Dict[str, asyncio.Future] = {}
        self._sse_client: Optional[httpx.AsyncClient] = None
        self._sse_task: Optional[asyncio.Task] = None

    def _make_headers(self, for_sse_get: bool = False) -> Dict[str, str]:
        headers = {}
        if not for_sse_get:
            headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json, text/event-stream"
        headers.update(self.config.headers)
        if self.session_id and not for_sse_get:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _jsonrpc_request(self, method: str, params: Optional[Dict] = None) -> Dict:
        req: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
        }
        if params is not None:
            req["params"] = params
        return req

    def _parse_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Parse response, handling both JSON and SSE formats."""
        content_type = response.headers.get("content-type", "")

        if "text/event-stream" in content_type:
            return self._parse_sse_text(response.text)
        else:
            return response.json()

    def _parse_sse_text(self, text: str) -> Dict[str, Any]:
        """Extract the last JSON-RPC response from SSE text."""
        result = None
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str:
                    try:
                        parsed = json.loads(data_str)
                        if isinstance(parsed, dict) and ("result" in parsed or "error" in parsed):
                            result = parsed
                    except json.JSONDecodeError:
                        continue
        if result:
            return result
        raise ValueError(f"No valid JSON-RPC response in SSE from {self.config.name}")

    def _base_url(self) -> str:
        """Get the base URL (without /mcp path) for SSE endpoint construction."""
        url = self.config.url
        for suffix in ("/mcp/", "/mcp"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url.rstrip("/")

    async def _start_sse_stream(self) -> str:
        """Start SSE stream and return the message endpoint URL."""
        base = self._base_url()
        sse_url = f"{base}/sse"

        self._sse_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
        )

        # We need to connect to SSE and get the endpoint
        # Use a streaming request
        self._sse_stream = await self._sse_client.send(
            self._sse_client.build_request(
                "GET", sse_url, headers=self._make_headers(for_sse_get=True)
            ),
            stream=True,
        )

        if self._sse_stream.status_code != 200:
            raise ConnectionError(
                f"SSE connection failed with status {self._sse_stream.status_code}"
            )

        # Read the initial endpoint event
        buffer = ""
        async for chunk in self._sse_stream.aiter_text():
            buffer += chunk
            if "endpoint" in buffer and "/messages/" in buffer:
                break
            if len(buffer) > 4096:
                break

        # Parse endpoint
        for line in buffer.split("\n"):
            line = line.strip()
            if line.startswith("data:") and "/messages/" in line:
                endpoint_path = line[5:].strip()
                message_url = f"{base}{endpoint_path}"
                logger.info(f"MCP '{self.config.name}': SSE message endpoint: {message_url}")

                # Start background task to read SSE responses
                self._sse_task = asyncio.create_task(self._read_sse_stream(buffer))
                return message_url

        raise ConnectionError(f"No endpoint event received from SSE stream at {sse_url}")

    async def _read_sse_stream(self, initial_buffer: str) -> None:
        """Background task to read SSE responses and resolve futures."""
        try:
            # Process any responses already in the initial buffer
            self._process_sse_buffer(initial_buffer)

            # Continue reading
            async for chunk in self._sse_stream.aiter_text():
                self._process_sse_buffer(chunk)
        except Exception as e:
            logger.debug(f"SSE stream ended for '{self.config.name}': {e}")
        finally:
            # Resolve any pending futures with errors
            for req_id, future in self._sse_responses.items():
                if not future.done():
                    future.set_exception(ConnectionError("SSE stream closed"))

    def _process_sse_buffer(self, text: str) -> None:
        """Process SSE text and resolve any pending futures."""
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str:
                    try:
                        parsed = json.loads(data_str)
                        if isinstance(parsed, dict) and "id" in parsed:
                            req_id = str(parsed["id"])
                            if req_id in self._sse_responses:
                                future = self._sse_responses.pop(req_id)
                                if not future.done():
                                    future.set_result(parsed)
                    except json.JSONDecodeError:
                        continue

    async def _post_jsonrpc(self, req: Dict) -> Dict[str, Any]:
        """Send a JSON-RPC request using the detected transport."""
        if self._transport == "sse":
            return await self._post_jsonrpc_sse(req)
        else:
            return await self._post_jsonrpc_streamable(req)

    async def _post_jsonrpc_streamable(self, req: Dict) -> Dict[str, Any]:
        """Send via Streamable HTTP (response in POST body)."""
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self.config.url, json=req, headers=self._make_headers()
            )
            resp.raise_for_status()
            return self._parse_response(resp)

    async def _post_jsonrpc_sse(self, req: Dict) -> Dict[str, Any]:
        """Send via SSE transport (response comes through event stream)."""
        if not self._sse_message_url:
            raise RuntimeError("SSE transport not initialized")

        req_id = str(req.get("id", ""))
        is_notification = "id" not in req  # Notifications don't get responses

        if not is_notification and req_id:
            # Create a future to wait for the response
            loop = asyncio.get_event_loop()
            future: asyncio.Future = loop.create_future()
            self._sse_responses[req_id] = future

        # POST the request (will get 202 Accepted)
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self._sse_message_url, json=req, headers=self._make_headers()
            )
            if resp.status_code not in (200, 202):
                resp.raise_for_status()

        if is_notification:
            return {}

        # Wait for the response from the SSE stream
        try:
            result = await asyncio.wait_for(future, timeout=120.0)
            return result
        except asyncio.TimeoutError:
            self._sse_responses.pop(req_id, None)
            raise TimeoutError(
                f"Timeout waiting for SSE response to {req.get('method')} "
                f"from '{self.config.name}'"
            )

    async def _detect_transport(self) -> str:
        """Detect which transport the server supports."""
        timeout = httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)

        # Try Streamable HTTP first
        try:
            req = self._jsonrpc_request("initialize", {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "jada-research", "version": "1.0.0"},
            })
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.config.url, json=req, headers=self._make_headers()
                )
                if resp.status_code == 200:
                    self._parse_response(resp)
                    if "mcp-session-id" in resp.headers:
                        self.session_id = resp.headers["mcp-session-id"]
                    logger.info(f"MCP '{self.config.name}': Streamable HTTP transport")
                    self._transport = "streamable_http"
                    return "streamable_http"
        except Exception as e:
            logger.debug(f"Streamable HTTP failed for '{self.config.name}': {e}")

        # Fall back to SSE
        try:
            self._sse_message_url = await self._start_sse_stream()
            self._transport = "sse"
            logger.info(f"MCP '{self.config.name}': SSE transport")
            return "sse"
        except Exception as e:
            logger.debug(f"SSE failed for '{self.config.name}': {e}")

        raise ConnectionError(
            f"MCP server '{self.config.name}' doesn't respond on "
            f"Streamable HTTP ({self.config.url}) or SSE ({self._base_url()}/sse)"
        )

    async def initialize(self) -> bool:
        """Perform MCP initialization handshake."""
        if self._initialized:
            return True

        try:
            transport = await self._detect_transport()

            if transport == "sse":
                # Send initialize via the SSE message endpoint
                req = self._jsonrpc_request("initialize", {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "jada-research", "version": "1.0.0"},
                })
                data = await self._post_jsonrpc(req)
                server_info = data.get("result", {}).get("serverInfo", {})
                logger.info(f"MCP '{self.config.name}' initialized (SSE). Server: {server_info}")

            # Send initialized notification
            notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
            try:
                await self._post_jsonrpc(notif)
            except Exception:
                pass  # Non-fatal

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP '{self.config.name}': {e}")
            self._initialized = False
            return False

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Discover available tools from the MCP server."""
        if not await self.initialize():
            return []

        try:
            req = self._jsonrpc_request("tools/list")
            data = await self._post_jsonrpc(req)
            tools = data.get("result", {}).get("tools", [])
            logger.info(f"Discovered {len(tools)} tools from '{self.config.name}'")
            return tools
        except Exception as e:
            logger.error(f"Failed to list tools from '{self.config.name}': {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not await self.initialize():
            raise RuntimeError(f"MCP server '{self.config.name}' not initialized")

        try:
            req = self._jsonrpc_request("tools/call", {
                "name": tool_name,
                "arguments": arguments,
            })
            data = await self._post_jsonrpc(req)

            result = data.get("result", {})
            if result.get("isError"):
                error_content = result.get("content", [{}])
                error_text = (
                    error_content[0].get("text", "Unknown error")
                    if error_content
                    else "Unknown error"
                )
                logger.warning(f"MCP tool '{tool_name}' error: {error_text}")
                return {"error": error_text}

            content = result.get("content", [])
            if content and isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts) if texts else str(content)
            return str(result)

        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed on '{self.config.name}': {e}")
            raise

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to the MCP server."""
        # Reset state
        self._initialized = False
        self.session_id = None
        self._transport = None
        self._sse_message_url = None
        await self._close_sse()

        try:
            success = await self.initialize()
            if success:
                tools = await self.list_tools()
                return {
                    "status": "connected",
                    "transport": self._transport,
                    "tool_count": len(tools),
                    "tools": [t.get("name", "unknown") for t in tools[:20]],
                }
            return {"status": "failed", "error": "Initialization failed"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
        finally:
            await self._close_sse()

    async def _close_sse(self) -> None:
        """Clean up SSE resources."""
        if self._sse_task and not self._sse_task.done():
            self._sse_task.cancel()
            self._sse_task = None
        if hasattr(self, "_sse_stream") and self._sse_stream:
            try:
                await self._sse_stream.aclose()
            except Exception:
                pass
            self._sse_stream = None
        if self._sse_client:
            try:
                await self._sse_client.aclose()
            except Exception:
                pass
            self._sse_client = None
        self._sse_responses.clear()
