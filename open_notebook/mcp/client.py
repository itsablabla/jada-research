"""
MCP client supporting both Streamable HTTP and SSE transports.

Transport detection:
- First tries Streamable HTTP (POST to URL with JSON-RPC)
- If that fails, falls back to SSE (GET /sse → POST /messages/?session_id=...)

SSE transport: Opens a persistent SSE stream per session. All operations
(initialize, initialized notification, tools/list, tools/call) go through
the same SSE session to maintain server-side state.
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
        # SSE session state
        self._sse_client: Optional[httpx.AsyncClient] = None
        self._sse_response: Optional[httpx.Response] = None
        self._sse_message_url: Optional[str] = None
        self._sse_buffer: str = ""
        self._sse_aiter: Optional[Any] = None

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

    def _jsonrpc_notification(self, method: str, params: Optional[Dict] = None) -> Dict:
        """Create a JSON-RPC notification (no id field)."""
        notif: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            notif["params"] = params
        return notif

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

    async def _ensure_sse_session(self) -> None:
        """Open an SSE session if one isn't already open."""
        if self._sse_message_url and self._sse_client:
            return  # Already have an active session

        await self._close_sse()  # Clean up any partial state

        base = self._base_url()
        sse_url = f"{base}/sse"
        timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)

        self._sse_client = httpx.AsyncClient(timeout=timeout)
        self._sse_response = await self._sse_client.send(
            self._sse_client.build_request(
                "GET", sse_url, headers=self._make_headers(for_sse_get=True)
            ),
            stream=True,
        )

        if self._sse_response.status_code != 200:
            status = self._sse_response.status_code
            await self._close_sse()
            raise ConnectionError(f"SSE connection failed: HTTP {status}")

        self._sse_aiter = self._sse_response.aiter_text()

        # Read until we get the endpoint event
        async for chunk in self._sse_aiter:
            self._sse_buffer += chunk
            while "\n" in self._sse_buffer:
                line, self._sse_buffer = self._sse_buffer.split("\n", 1)
                line = line.strip()
                if line.startswith("data:") and "/messages/" in line:
                    endpoint_path = line[5:].strip()
                    self._sse_message_url = f"{base}{endpoint_path}"
                    logger.debug(f"SSE endpoint: {self._sse_message_url}")
                    return

        await self._close_sse()
        raise ConnectionError(f"No endpoint event from SSE at {sse_url}")

    async def _sse_post(self, req: Dict) -> None:
        """POST a JSON-RPC request/notification to the SSE message endpoint."""
        if not self._sse_client or not self._sse_message_url:
            raise RuntimeError("No active SSE session")

        resp = await self._sse_client.post(
            self._sse_message_url, json=req, headers=self._make_headers()
        )
        if resp.status_code not in (200, 202):
            raise ConnectionError(
                f"SSE POST failed: HTTP {resp.status_code} {resp.text}"
            )

    async def _sse_read_response(self, req_id: str) -> Dict[str, Any]:
        """Read from the SSE stream until we get a response matching req_id."""
        if not self._sse_aiter:
            raise RuntimeError("No active SSE stream")

        async for chunk in self._sse_aiter:
            self._sse_buffer += chunk
            while "\n" in self._sse_buffer:
                line, self._sse_buffer = self._sse_buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str:
                    continue
                try:
                    parsed = json.loads(data_str)
                    if isinstance(parsed, dict) and (
                        "result" in parsed or "error" in parsed
                    ):
                        if str(parsed.get("id", "")) == req_id:
                            return parsed
                except json.JSONDecodeError:
                    continue

        raise TimeoutError(f"SSE stream closed before response for {req_id}")

    async def _sse_request(self, req: Dict) -> Dict[str, Any]:
        """Send a JSON-RPC request over the persistent SSE session."""
        await self._ensure_sse_session()
        req_id = str(req.get("id", ""))
        await self._sse_post(req)
        if "id" not in req:
            return {}  # Notification, no response expected
        return await self._sse_read_response(req_id)

    async def _sse_notify(self, notif: Dict) -> None:
        """Send a JSON-RPC notification over the persistent SSE session."""
        await self._ensure_sse_session()
        await self._sse_post(notif)

    async def _close_sse(self) -> None:
        """Close the SSE session and clean up resources."""
        if self._sse_response:
            try:
                await self._sse_response.aclose()
            except Exception:
                pass
            self._sse_response = None
        if self._sse_client:
            try:
                await self._sse_client.aclose()
            except Exception:
                pass
            self._sse_client = None
        self._sse_message_url = None
        self._sse_buffer = ""
        self._sse_aiter = None

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

        Tries Streamable HTTP first. If that fails, falls back to SSE.
        For SSE: opens persistent session, sends initialize, sends initialized
        notification — all in the same session.
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
                    logger.info(
                        f"MCP '{self.config.name}' initialized (Streamable HTTP)"
                    )
                    return True
        except Exception as e:
            logger.debug(f"Streamable HTTP failed for '{self.config.name}': {e}")

        # --- Fall back to SSE ---
        try:
            # Open persistent SSE session
            await self._ensure_sse_session()

            # Send initialize request
            req = self._jsonrpc_request("initialize", init_params)
            data = await self._sse_request(req)
            server_info = data.get("result", {}).get("serverInfo", {})
            logger.info(
                f"MCP '{self.config.name}' init response (SSE). "
                f"Server: {server_info}"
            )

            # Send initialized notification (required by MCP protocol)
            notif = self._jsonrpc_notification("notifications/initialized")
            await self._sse_notify(notif)
            logger.debug(f"MCP '{self.config.name}': sent initialized notification")

            self._transport = "sse"
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP '{self.config.name}': {e}")
            await self._close_sse()
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
        # Reset state for a fresh test
        self._initialized = False
        self.session_id = None
        self._transport = None
        await self._close_sse()

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
        finally:
            # Close SSE session after test to free resources
            await self._close_sse()
            self._initialized = False
