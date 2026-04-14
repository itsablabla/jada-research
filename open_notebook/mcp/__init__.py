"""
MCP (Model Context Protocol) client integration for Open Notebook.
Enables the chat model to call tools from external MCP servers.
"""

from open_notebook.mcp.client import MCPClient
from open_notebook.mcp.config import MCPConfigManager
from open_notebook.mcp.langchain_bridge import get_mcp_langchain_tools

__all__ = ["MCPClient", "MCPConfigManager", "get_mcp_langchain_tools"]
