"""
MCP client supporting both Streamable HTTP and SSE transports.
Connects to MCP servers, performs initialization handshake, and discovers tools.

Transport detection:
- First tries Streamable HTTP (POST to URL with JSON-RPC)
- If that returns 404/405, falls back to SSE (GET /sse → POST /messages/?session_id=...)
"""

import json
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

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
        self._sse_message_url: Optional[str] = None  # For SSE: the POST endpoint

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
            # Parse SSE: extract JSON from data: lines
            result = None
            for line in response.text.split("\n"):
                line = line.strip()
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str:
                        try:
                            parsed = json.loads(data_str)
                            # Only accept JSON-RPC responses (with "result" or "error")
                            if isinstance(parsed, dict) and ("result" in parsed or "error" in parsed):
                                result = parsed
                        except json.JSONDecodeError:
                            continue
            if result:
                return result
            raise ValueError(f"No valid JSON-RPC response in SSE from {self.config.name}")
        else:
            return response.json()

    def _base_url(self) -> str:
        """Get the base URL (without /mcp path) for SSE endpoint construction."""
        url = self.config.url
        # Remove trailing /mcp or /mcp/ to get base
        for suffix in ("/mcp/", "/mcp"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url.rstrip("/")

    async def _detect_transport(self) -> str:
        """Detect which transport the server supports."""
        timeout = httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)

        # Try Streamable HTTP first (POST to the configured URL)
        try:
            req = self._jsonrpc_request("initialize", {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "jada-research", "version": "1.0.0"},
            })
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.config.url,
                    json=req,
                    headers=self._make_headers(),
                )
                if resp.status_code == 200:
                    data = self._parse_response(resp)
                    if "mcp-session-id" in resp.headers:
                        self.session_id = resp.headers["mcp-session-id"]
                    logger.info(f"MCP '{self.config.name}': using Streamable HTTP transport")
                    self._transport = "streamable_http"
                    return "streamable_http"
        except Exception as e:
            logger.debug(f"Streamable HTTP failed for '{self.config.name}': {e}")

        # Fall back to SSE transport
        try:
            base = self._base_url()
            sse_url = f"{base}/sse"
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "GET", sse_url, headers=self._make_headers(for_sse_get=True)
                ) as resp:
                    if resp.status_code == 200:
                        # Read just enough to get the endpoint event
                        buffer = ""
                        async for chunk in resp.aiter_text():
                            buffer += chunk
                            if "endpoint" in buffer and "data:" in buffer:
                                break
                            if len(buffer) > 4096:
                                break

                        # Parse the endpoint from SSE
                        for line in buffer.split("\n"):
                            line = line.strip()
                            if line.startswith("data:") and "/messages/" in line:
                                endpoint_path = line[5:].strip()
                                self._sse_message_url = f"{base}{endpoint_path}"
                                self._transport = "sse"
                                logger.info(
                                    f"MCP '{self.config.name}': using SSE transport, "
                                    f"message endpoint: {self._sse_message_url}"
                                )
                                return "sse"
        except Exception as e:
            logger.debug(f"SSE detection failed for '{self.config.name}': {e}")

        raise ConnectionError(
            f"MCP server '{self.config.name}' doesn't respond on either "
            f"Streamable HTTP ({self.config.url}) or SSE ({self._base_url()}/sse)"
        )

    async def _post_jsonrpc(self, req: Dict) -> Dict[str, Any]:
        """Send a JSON-RPC request using the detected transport."""
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

        if self._transport == "sse":
            if not self._sse_message_url:
                raise RuntimeError("SSE transport not initialized (no message URL)")
            url = self._sse_message_url
        else:
            url = self.config.url

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=req, headers=self._make_headers())
            resp.raise_for_status()
            return self._parse_response(resp)

    async def initialize(self) -> bool:
        """Perform MCP initialization handshake."""
        if self._initialized:
            return True

        try:
            # Detect transport (this also sends initialize for streamable_http)
            transport = await self._detect_transport()

            if transport == "sse":
                # For SSE, we need to send initialize via the message endpoint
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
                pass  # Notification failures are non-fatal

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP server '{self.config.name}': {e}")
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
                logger.warning(f"MCP tool '{tool_name}' returned error: {error_text}")
                return {"error": error_text}

            # Extract text content from result
            content = result.get("content", [])
            if content and isinstance(content, list):
                texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                return "\n".join(texts) if texts else str(content)
            return str(result)

        except Exception as e:
            logger.error(f"Failed to call tool '{tool_name}' on '{self.config.name}': {e}")
            raise

    async def test_connection(self) -> Dict[str, Any]:
        """Test the connection to the MCP server."""
        self._initialized = False
        self.session_id = None
        self._transport = None
        self._sse_message_url = None
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
