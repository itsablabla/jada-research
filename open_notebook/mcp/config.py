"""
MCP server configuration management.
Stores and retrieves MCP server configs from a JSON file in the data directory.
"""

import json
import os
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from open_notebook.config import DATA_FOLDER


def _encrypt_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Encrypt header values before storing to disk."""
    if not headers:
        return headers
    try:
        from open_notebook.utils.encryption import encrypt_value, looks_like_fernet_token
        return {
            k: (v if looks_like_fernet_token(v) else encrypt_value(v))
            for k, v in headers.items()
        }
    except (ValueError, ImportError):
        # Encryption not configured — store as-is (dev mode)
        return headers


def _decrypt_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Decrypt header values after reading from disk."""
    if not headers:
        return headers
    try:
        from open_notebook.utils.encryption import decrypt_value
        return {k: decrypt_value(v) for k, v in headers.items()}
    except (ValueError, ImportError):
        return headers

MCP_CONFIG_FILE = os.path.join(DATA_FOLDER, "mcp_servers.json")


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    id: str = Field(..., description="Unique identifier for this server")
    name: str = Field(..., description="Human-readable name")
    url: str = Field(..., description="MCP server URL (Streamable HTTP endpoint)")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="HTTP headers (e.g. Authorization)"
    )
    enabled: bool = Field(default=True, description="Whether this server is active")
    description: Optional[str] = Field(
        None, description="Optional description of what this server provides"
    )


class MCPConfigManager:
    """Manages MCP server configurations persisted to disk."""

    def __init__(self, config_file: str = MCP_CONFIG_FILE):
        self.config_file = config_file

    def _load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.config_file):
            return []
        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading MCP config: {e}")
            return []

    def _save(self, servers: List[Dict[str, Any]]) -> None:
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, "w") as f:
            json.dump(servers, f, indent=2)

    def list_servers(self) -> List[MCPServerConfig]:
        """List all configured MCP servers (decrypts headers)."""
        raw = self._load()
        for s in raw:
            s["headers"] = _decrypt_headers(s.get("headers", {}))
        return [MCPServerConfig(**s) for s in raw]

    def get_server(self, server_id: str) -> Optional[MCPServerConfig]:
        """Get a specific server by ID (decrypts headers)."""
        for s in self._load():
            if s.get("id") == server_id:
                s["headers"] = _decrypt_headers(s.get("headers", {}))
                return MCPServerConfig(**s)
        return None

    def add_server(self, config: MCPServerConfig) -> MCPServerConfig:
        """Add a new MCP server configuration."""
        servers = self._load()
        # Check for duplicate ID
        for s in servers:
            if s.get("id") == config.id:
                raise ValueError(f"Server with ID '{config.id}' already exists")
        data = config.model_dump()
        data["headers"] = _encrypt_headers(data.get("headers", {}))
        servers.append(data)
        self._save(servers)
        logger.info(f"Added MCP server: {config.name} ({config.id})")
        return config

    def update_server(self, server_id: str, updates: Dict[str, Any]) -> Optional[MCPServerConfig]:
        """Update an existing MCP server configuration."""
        servers = self._load()
        for i, s in enumerate(servers):
            if s.get("id") == server_id:
                # Encrypt new headers if provided
                if "headers" in updates and updates["headers"]:
                    updates["headers"] = _encrypt_headers(updates["headers"])
                s.update(updates)
                s["id"] = server_id  # Prevent ID change
                servers[i] = s
                self._save(servers)
                logger.info(f"Updated MCP server: {server_id}")
                # Return with decrypted headers
                s["headers"] = _decrypt_headers(s.get("headers", {}))
                return MCPServerConfig(**s)
        return None

    def delete_server(self, server_id: str) -> bool:
        """Delete an MCP server configuration."""
        servers = self._load()
        new_servers = [s for s in servers if s.get("id") != server_id]
        if len(new_servers) == len(servers):
            return False
        self._save(new_servers)
        logger.info(f"Deleted MCP server: {server_id}")
        return True

    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get only enabled MCP servers."""
        return [s for s in self.list_servers() if s.enabled]


# Global instance
mcp_config_manager = MCPConfigManager()
