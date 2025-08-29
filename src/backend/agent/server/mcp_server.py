"""
File: agent/server/mcp_server.py

This file contains the implementation of the MCP (Model Context Protocol) server.
It will maintain the connection of 
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")

@mcp.tool()