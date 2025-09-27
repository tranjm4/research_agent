"""
File: src/backend/agent/servers/documents_server.py

This is the MCP server that interacts with the vector database for internally stored research documents
"""

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("documents-server")

@mcp.resource(uri="mcp://documents/v1/papers",
              name="list-tools",
              title="Get Tools",
              description="Provides the Host with the available tools and resources")
def get_tools_and_resources():
    pass