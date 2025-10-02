"""
File: src/backend/agent/servers/documents_server.py

This is the MCP server that interacts with the vector database for internally stored research documents
"""

from mcp.server.fastmcp import FastMCP
from vectorstore import MongoVectorStore

from typing import List, Dict, Any, Optional
import logging

import os
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

mcp = FastMCP("documents-server")

# Initialize vectorstore upon server startup
MONGO_URI = os.getenv("MONGO_URI")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
DB_NAME = os.getenv("DB_NAME")  # Fixed: was CHUNKS_COLLECTION_NAME

if not MONGO_URI or not DB_NAME or not COLLECTION_NAME:
    raise ValueError("Missing required environment variables: MONGO_URI, DB_NAME, or COLLECTION_NAME")

try:
    vectorstore = MongoVectorStore(
        mongo_uri=MONGO_URI,
        db_name=DB_NAME,
        collection_name=COLLECTION_NAME
    )
    logger.info("Vectorstore initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize vectorstore: {e}")
    raise

@mcp.resource(uri="mcp://documents/v1/papers",
              name="list-tools",
              title="Get Tools",
              description="Provides the Host with the available tools and resources")
def get_tools_and_resources():
    pass



@mcp.tool()
async def search_documents(query: str, top_k: int = 5, filter_topics: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search internal research documents using vector similarity

    Args:
        query: The search query text
        top_k: Number of top results to return (default: 5)
        filter_topics: Optional topic filter (e.g., 'cs.AI', 'machine learning')

    Returns:
        List of search results with scores, text chunks, and metadata
    """
    filter_by = {'topics': filter_topics} if filter_topics else None
    results = vectorstore.search(query, top_k=top_k, filter_by=filter_by)
    return results


@mcp.tool()
async def get_paper_chunks(paper_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve all text chunks from a specific paper

    Args:
        paper_id: The paper ID to retrieve chunks for
        limit: Maximum number of chunks to return (default: 50)

    Returns:
        List of text chunks with metadata for the specified paper
    """
    return vectorstore.search_by_paper(paper_id, limit=limit)


@mcp.tool()
async def find_papers_by_topic(topic: str, limit: int = 20) -> List[Dict[str, str]]:
    """
    Find papers that match a specific topic

    Args:
        topic: Topic to search for (case-insensitive, e.g., 'machine learning', 'cs.AI')
        limit: Maximum number of papers to return (default: 20)

    Returns:
        List of unique papers with metadata (title, authors, url, topics, keywords)
    """
    return vectorstore.get_papers_by_topic(topic, limit=limit)


@mcp.tool()
async def get_vectorstore_stats() -> Dict[str, Any]:
    """
    Get statistics about the vectorstore

    Returns:
        Dictionary with total chunks, papers, topics, and index info
    """
    return vectorstore.get_stats()


if __name__ == "__main__":
    mcp.run()