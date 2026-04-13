"""
MCP client for Streamable HTTP transport.
Connects to MCP servers, performs initialization handshake, and discovers tools.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from open_notebook.mcp.config import MCPServerConfig


class MCPClient:
    """Client for communicating with MCP servers via Streamable HTTP."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session_id: Optional[str] = None
        self._initialized = False

    def _make_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        headers.update(self.config.headers)
        if self.session_id:
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
                            result = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
            if result:
                return result
            raise ValueError(f"No valid JSON found in SSE response from {self.config.name}")
        else:
            return response.json()

    async def initialize(self) -> bool:
        """Perform MCP initialization handshake."""
        if self._initialized:
            return True

        try:
            req = self._jsonrpc_request("initialize", {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "jada-research",
                    "version": "1.0.0",
                },
            })

            timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.config.url,
                    json=req,
                    headers=self._make_headers(),
                )
                resp.raise_for_status()

                data = self._parse_response(resp)

                # Capture session ID if provided
                if "mcp-session-id" in resp.headers:
                    self.session_id = resp.headers["mcp-session-id"]

                # Send initialized notification
                notif = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                }
                await client.post(
                    self.config.url,
                    json=notif,
                    headers=self._make_headers(),
                )

                self._initialized = True
                logger.info(
                    f"MCP server '{self.config.name}' initialized. "
                    f"Server: {data.get('result', {}).get('serverInfo', {})}"
                )
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
            timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.config.url,
                    json=req,
                    headers=self._make_headers(),
                )
                resp.raise_for_status()
                data = self._parse_response(resp)
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

            timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    self.config.url,
                    json=req,
                    headers=self._make_headers(),
                )
                resp.raise_for_status()
                data = self._parse_response(resp)

                result = data.get("result", {})
                if result.get("isError"):
                    error_content = result.get("content", [{}])
                    error_text = error_content[0].get("text", "Unknown error") if error_content else "Unknown error"
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
        try:
            success = await self.initialize()
            if success:
                tools = await self.list_tools()
                return {
                    "status": "connected",
                    "tool_count": len(tools),
                    "tools": [t.get("name", "unknown") for t in tools[:20]],
                }
            return {"status": "failed", "error": "Initialization failed"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
