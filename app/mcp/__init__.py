"""
MCP (Model Context Protocol) server module.
Exposes tools for code exploration and analysis.
"""
from .server import mcp, get_mcp_server

__all__ = [
    "mcp",
    "get_mcp_server",
]
