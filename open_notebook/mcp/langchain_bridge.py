"""
Bridge between MCP tools and LangChain tools.
Converts MCP tool definitions into LangChain-compatible tools that can be
bound to chat models for tool calling.
"""

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional, Type

from langchain_core.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, Field, create_model

from open_notebook.mcp.client import MCPClient
from open_notebook.mcp.config import MCPServerConfig, mcp_config_manager

# Cache of MCP clients (keyed by server ID)
_client_cache: Dict[str, MCPClient] = {}


def _get_client(config: MCPServerConfig) -> MCPClient:
    """Get or create a cached MCP client for a server config."""
    if config.id not in _client_cache:
        _client_cache[config.id] = MCPClient(config)
    return _client_cache[config.id]


def _json_schema_to_pydantic_field(
    name: str, schema: Dict[str, Any], required: bool
) -> tuple:
    """Convert a JSON Schema property to a Pydantic field tuple."""
    field_type: Any = str  # default
    json_type = schema.get("type", "string")

    if json_type == "string":
        field_type = str
    elif json_type == "integer":
        field_type = int
    elif json_type == "number":
        field_type = float
    elif json_type == "boolean":
        field_type = bool
    elif json_type == "array":
        field_type = list
    elif json_type == "object":
        field_type = dict
    else:
        field_type = str

    description = schema.get("description", "")
    default = ... if required else schema.get("default", None)

    if not required:
        field_type = Optional[field_type]

    return (field_type, Field(default=default, description=description))


def _build_args_model(tool_def: Dict[str, Any]) -> Type[BaseModel]:
    """Build a Pydantic model from an MCP tool's inputSchema."""
    input_schema = tool_def.get("inputSchema", {})
    properties = input_schema.get("properties", {})
    required_fields = set(input_schema.get("required", []))

    if not properties:
        # No parameters — create an empty model
        return create_model(f"{tool_def['name']}_Args")

    fields = {}
    for prop_name, prop_schema in properties.items():
        is_required = prop_name in required_fields
        fields[prop_name] = _json_schema_to_pydantic_field(
            prop_name, prop_schema, is_required
        )

    return create_model(f"{tool_def['name']}_Args", **fields)


def _make_tool_func(client: MCPClient, tool_name: str) -> Callable:
    """Create a callable that invokes an MCP tool."""

    def tool_func(**kwargs: Any) -> str:
        """Execute an MCP tool call."""
        try:
            # Run async call_tool in a new event loop (we're called from sync context)
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(client.call_tool(tool_name, kwargs))
            finally:
                loop.close()

            if isinstance(result, dict) and "error" in result:
                return f"Error: {result['error']}"
            if isinstance(result, str):
                return result
            return json.dumps(result, default=str)
        except Exception as e:
            logger.error(f"MCP tool '{tool_name}' execution failed: {e}")
            return f"Error executing tool: {e}"

    return tool_func


async def discover_server_tools(config: MCPServerConfig) -> List[Dict[str, Any]]:
    """Discover tools from a single MCP server."""
    client = _get_client(config)
    return await client.list_tools()


async def get_mcp_langchain_tools() -> List[StructuredTool]:
    """
    Get all MCP tools from enabled servers as LangChain StructuredTools.
    This is the main entry point for the chat graph to get available tools.
    """
    tools: List[StructuredTool] = []
    enabled_servers = mcp_config_manager.get_enabled_servers()

    if not enabled_servers:
        return tools

    for server_config in enabled_servers:
        try:
            client = _get_client(server_config)
            mcp_tools = await client.list_tools()

            for tool_def in mcp_tools:
                try:
                    name = tool_def.get("name", "unknown")
                    description = tool_def.get("description", f"MCP tool: {name}")
                    # Prefix with server name to avoid collisions
                    prefixed_name = f"{server_config.id}__{name}"

                    args_model = _build_args_model(tool_def)
                    func = _make_tool_func(client, name)

                    lc_tool = StructuredTool(
                        name=prefixed_name,
                        description=f"[{server_config.name}] {description}",
                        func=func,
                        args_schema=args_model,
                    )
                    tools.append(lc_tool)
                except Exception as e:
                    logger.warning(
                        f"Failed to convert MCP tool '{tool_def.get('name')}' "
                        f"from '{server_config.name}': {e}"
                    )
                    continue

            logger.info(
                f"Loaded {len(mcp_tools)} tools from MCP server '{server_config.name}'"
            )

        except Exception as e:
            logger.error(
                f"Failed to load tools from MCP server '{server_config.name}': {e}"
            )
            continue

    logger.info(f"Total MCP tools available: {len(tools)}")
    return tools


def clear_client_cache() -> None:
    """Clear the MCP client cache (e.g. after config changes)."""
    _client_cache.clear()
