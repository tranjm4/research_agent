"""
File: src/backend/agent/servers/application_server.py

This is the MCP server that interacts with the application database (user's documents, conversations, etc.)
"""

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("application-server")

@mcp.resource(uri="mcp://documents/v1/papers",
              name="list-tools",
              title="Get Tools",
              description="Provides the Host with the available tools and resources")
def get_tools_and_resources():
    pass