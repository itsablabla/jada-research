"""
API router for MCP server management.
CRUD operations for MCP server configurations + tool discovery + connection testing.
"""

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from open_notebook.mcp.client import MCPClient
from open_notebook.mcp.config import MCPServerConfig, mcp_config_manager
from open_notebook.mcp.langchain_bridge import clear_client_cache

router = APIRouter()


# Request/Response models
class AddMCPServerRequest(BaseModel):
    name: str = Field(..., description="Human-readable server name")
    url: str = Field(..., description="MCP server URL (Streamable HTTP endpoint)")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="HTTP headers (e.g. Authorization)"
    )
    enabled: bool = Field(default=True, description="Whether this server is active")
    description: Optional[str] = Field(
        None, description="Optional description"
    )


class UpdateMCPServerRequest(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None


class MCPServerResponse(BaseModel):
    id: str
    name: str
    url: str
    headers: Dict[str, str]
    enabled: bool
    description: Optional[str] = None


class MCPToolInfo(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


class MCPConnectionTestResponse(BaseModel):
    status: str
    tool_count: int = 0
    tools: List[str] = Field(default_factory=list)
    error: Optional[str] = None


@router.get("/mcp/servers", response_model=List[MCPServerResponse])
async def list_servers():
    """List all configured MCP servers."""
    servers = mcp_config_manager.list_servers()
    return [
        MCPServerResponse(
            id=s.id,
            name=s.name,
            url=s.url,
            headers=s.headers,
            enabled=s.enabled,
            description=s.description,
        )
        for s in servers
    ]


@router.post("/mcp/servers", response_model=MCPServerResponse)
async def add_server(request: AddMCPServerRequest):
    """Add a new MCP server configuration."""
    try:
        server_id = f"mcp_{uuid.uuid4().hex[:12]}"
        config = MCPServerConfig(
            id=server_id,
            name=request.name,
            url=request.url,
            headers=request.headers,
            enabled=request.enabled,
            description=request.description,
        )
        result = mcp_config_manager.add_server(config)
        clear_client_cache()
        return MCPServerResponse(
            id=result.id,
            name=result.name,
            url=result.url,
            headers=result.headers,
            enabled=result.enabled,
            description=result.description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error adding MCP server: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding MCP server: {e}")


@router.get("/mcp/servers/{server_id}", response_model=MCPServerResponse)
async def get_server(server_id: str):
    """Get a specific MCP server configuration."""
    server = mcp_config_manager.get_server(server_id)
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")
    return MCPServerResponse(
        id=server.id,
        name=server.name,
        url=server.url,
        headers=server.headers,
        enabled=server.enabled,
        description=server.description,
    )


@router.put("/mcp/servers/{server_id}", response_model=MCPServerResponse)
async def update_server(server_id: str, request: UpdateMCPServerRequest):
    """Update an MCP server configuration."""
    updates = request.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    result = mcp_config_manager.update_server(server_id, updates)
    if not result:
        raise HTTPException(status_code=404, detail="MCP server not found")

    clear_client_cache()
    return MCPServerResponse(
        id=result.id,
        name=result.name,
        url=result.url,
        headers=result.headers,
        enabled=result.enabled,
        description=result.description,
    )


@router.delete("/mcp/servers/{server_id}")
async def delete_server(server_id: str):
    """Delete an MCP server configuration."""
    success = mcp_config_manager.delete_server(server_id)
    if not success:
        raise HTTPException(status_code=404, detail="MCP server not found")
    clear_client_cache()
    return {"success": True, "message": f"Server {server_id} deleted"}


@router.post("/mcp/servers/{server_id}/test", response_model=MCPConnectionTestResponse)
async def test_connection(server_id: str):
    """Test connection to an MCP server and discover its tools."""
    server = mcp_config_manager.get_server(server_id)
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")

    client = MCPClient(server)
    result = await client.test_connection()

    return MCPConnectionTestResponse(
        status=result.get("status", "unknown"),
        tool_count=result.get("tool_count", 0),
        tools=result.get("tools", []),
        error=result.get("error"),
    )


@router.get("/mcp/servers/{server_id}/tools", response_model=List[MCPToolInfo])
async def list_server_tools(server_id: str):
    """List all tools available from an MCP server."""
    server = mcp_config_manager.get_server(server_id)
    if not server:
        raise HTTPException(status_code=404, detail="MCP server not found")

    client = MCPClient(server)
    tools = await client.list_tools()

    return [
        MCPToolInfo(
            name=t.get("name", "unknown"),
            description=t.get("description"),
            input_schema=t.get("inputSchema"),
        )
        for t in tools
    ]
