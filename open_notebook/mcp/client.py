"""
MCP client supporting both Streamable HTTP and SSE transports.

Transport detection:
- First tries Streamable HTTP (POST to URL with JSON-RPC)
- If that fails, falls back to SSE (GET /sse → POST /messages/?session_id=...)

SSE transport: Each operation opens a fresh SSE stream, gets the session endpoint,
sends the JSON-RPC request, reads the response from the stream, then closes.
This avoids the complexity of long-lived SSE connections across async contexts.
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
            return self._extract_jsonrpc_from_sse(response.text)
        else:
            return response.json()

    def _extract_jsonrpc_from_sse(self, text: str) -> Dict[str, Any]:
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
        raise ValueError(f"No JSON-RPC response in SSE text from {self.config.name}")

    def _base_url(self) -> str:
        """Get base URL (without /mcp suffix) for SSE endpoint."""
        url = self.config.url
        for suffix in ("/mcp/", "/mcp"):
            if url.endswith(suffix):
                return url[: -len(suffix)]
        return url.rstrip("/")

    async def _sse_request(self, req: Dict) -> Dict[str, Any]:
        """Execute a JSON-RPC request over SSE transport.

        Opens a fresh SSE stream, gets the session endpoint, POSTs the request,
        reads the response from the SSE stream, then closes everything.
        """
        base = self._base_url()
        sse_url = f"{base}/sse"
        req_id = str(req.get("id", ""))
        is_notification = "id" not in req

        timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

        async with httpx.AsyncClient(timeout=timeout) as client:
            # Step 1: Open SSE stream and get the message endpoint
            async with client.stream(
                "GET", sse_url, headers=self._make_headers(for_sse_get=True)
            ) as sse_resp:
                if sse_resp.status_code != 200:
                    raise ConnectionError(
                        f"SSE connection failed: HTTP {sse_resp.status_code}"
                    )

                # Read until we get the endpoint event
                message_url = None
                buffer = ""
                async for chunk in sse_resp.aiter_text():
                    buffer += chunk
                    for line in buffer.split("\n"):
                        line = line.strip()
                        if line.startswith("data:") and "/messages/" in line:
                            endpoint_path = line[5:].strip()
                            message_url = f"{base}{endpoint_path}"
                            break
                    if message_url:
                        break
                    if len(buffer) > 8192:
                        break

                if not message_url:
                    raise ConnectionError(
                        f"No endpoint event from SSE at {sse_url}"
                    )

                logger.debug(f"SSE endpoint for '{self.config.name}': {message_url}")

                # Step 2: POST the JSON-RPC request to the message endpoint
                post_resp = await client.post(
                    message_url, json=req, headers=self._make_headers()
                )
                if post_resp.status_code not in (200, 202):
                    raise ConnectionError(
                        f"SSE POST failed: HTTP {post_resp.status_code} "
                        f"{post_resp.text}"
                    )

                if is_notification:
                    return {}

                # Step 3: Read the response from the SSE stream
                # The response comes as a data: event with matching id
                response_buffer = ""
                async for chunk in sse_resp.aiter_text():
                    response_buffer += chunk
                    # Check if we have a complete JSON-RPC response
                    for line in response_buffer.split("\n"):
                        line = line.strip()
                        if line.startswith("data:"):
                            data_str = line[5:].strip()
                            if data_str:
                                try:
                                    parsed = json.loads(data_str)
                                    if isinstance(parsed, dict) and (
                                        "result" in parsed or "error" in parsed
                                    ):
                                        # Check if it matches our request id
                                        if str(parsed.get("id", "")) == req_id:
                                            return parsed
                                except json.JSONDecodeError:
                                    continue

        raise TimeoutError(
            f"No SSE response for request {req_id} from '{self.config.name}'"
        )

    async def _post_jsonrpc(self, req: Dict) -> Dict[str, Any]:
        """Send JSON-RPC request using the detected transport."""
        if self._transport == "sse":
            return await self._sse_request(req)

        # Streamable HTTP
        timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                self.config.url, json=req, headers=self._make_headers()
            )
            resp.raise_for_status()
            return self._parse_response(resp)

    async def initialize(self) -> bool:
        """Perform MCP initialization handshake.

        Tries Streamable HTTP first. If that fails (404, connection error, etc.),
        falls back to SSE transport and initializes through a fresh SSE session.
        """
        if self._initialized:
            return True

        init_params = {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "jada-research", "version": "1.0.0"},
        }

        # --- Try Streamable HTTP ---
        try:
            req = self._jsonrpc_request("initialize", init_params)
            timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.config.url, json=req, headers=self._make_headers()
                )
                if resp.status_code == 200:
                    self._parse_response(resp)
                    if "mcp-session-id" in resp.headers:
                        self.session_id = resp.headers["mcp-session-id"]
                    self._transport = "streamable_http"
                    self._initialized = True
                    logger.info(f"MCP '{self.config.name}' initialized (Streamable HTTP)")
                    return True
        except Exception as e:
            logger.debug(f"Streamable HTTP failed for '{self.config.name}': {e}")

        # --- Fall back to SSE: initialize in one shot ---
        try:
            req = self._jsonrpc_request("initialize", init_params)
            data = await self._sse_request(req)
            server_info = data.get("result", {}).get("serverInfo", {})
            self._transport = "sse"
            self._initialized = True
            logger.info(f"MCP '{self.config.name}' initialized (SSE). Server: {server_info}")
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
        self._initialized = False
        self.session_id = None
        self._transport = None

        try:
            success = await self.initialize()
            if success:
                tools = await self.list_tools()
                return {
                    "status": "connected",
                    "transport": self._transport,
                    "tool_count": len(tools),
                    "tools": [t.get("name", "unknown") for t in tools[:50]],
                }
            return {"status": "failed", "error": "Initialization failed"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
