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
async def search_internal_research_documents(query: str, top_k: int = 3, filter_topics: Optional[str] = None, use_hybrid: bool = False) -> str:
    """
    Search internal research documents using vector similarity or hybrid search

    Args:
        query: The search query text
        top_k: Number of top results to return (default: 3)
        filter_topics: Optional topic filter (e.g., 'cs.AI', 'machine learning')
        use_hybrid: Use hybrid search (vector + keyword) with reranking (default: False)

    Returns:
        Formatted string with relevant papers, citations, and metadata
    """
    filter_by = {'topics': filter_topics} if filter_topics else None

    if use_hybrid:
        results = vectorstore.hybrid_search(query, top_k=top_k, filter_by=filter_by)
    else:
        results = vectorstore.search(query, top_k=top_k, filter_by=filter_by)

    if not results:
        return "No relevant documents found in the internal database."

    # Log what we received for debugging
    logger.info(f"Vectorstore returned {len(results)} results")
    if results:
        first_result = results[0]
        logger.info(f"Sample result structure: keys={list(first_result.keys())}")
        logger.info(f"Sample text length: {len(first_result.get('text', ''))}")
        if 'metadata' in first_result:
            logger.info(f"Sample metadata keys: {list(first_result['metadata'].keys())}")

    # Format results concisely: abstract text (truncated) + citation
    formatted_results = []
    for i, result in enumerate(results, 1):
        text = result.get('text', '')
        metadata = result.get('metadata', {})

        # Truncate text to 800 chars (abstracts are longer than chunks were)
        abstract_preview = text[:800] + '...' if len(text) > 800 else text

        formatted_results.append(
            f"[{i}] Score: {result.get('score', 0):.3f}\n"
            f"Abstract: {abstract_preview}\n"
            f"Title: {metadata.get('title', 'Unknown')}\n"
            f"Authors: {', '.join(metadata.get('authors', []))[:100] if isinstance(metadata.get('authors'), list) else 'Unknown'}\n"
            f"URL: {metadata.get('url', 'N/A')}\n"
            f"Topics: {', '.join(metadata.get('topics', []))[:100] if isinstance(metadata.get('topics'), list) else 'N/A'}\n"
            f"Keywords: {', '.join(metadata.get('keywords', []))[:100] if isinstance(metadata.get('keywords'), list) else 'N/A'}"
        )
        logger.info(formatted_results[-1])
    return "\n\n---\n\n".join(formatted_results)


@mcp.tool()
async def get_paper_by_id(paper_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific paper by its ID

    Args:
        paper_id: The paper ID to retrieve

    Returns:
        Paper document with abstract, title, authors, keywords, and metadata
    """
    results = vectorstore.search_by_paper(paper_id, limit=1)
    if results:
        return results[0]
    else:
        return {"error": f"Paper {paper_id} not found"}


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
        Dictionary with total documents, papers, topics, and index info
    """
    return vectorstore.get_stats()


if __name__ == "__main__":
    mcp.run()