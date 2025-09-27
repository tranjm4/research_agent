"""
File: src/backend/agent/servers/search_server.py

This is the MCP server that interacts with the internet to search for relevant information.
"""

from mcp.server.fastmcp import FastMCP
import httpx
from typing import List, Dict, Any
import re
from ddgs import DDGS

mcp = FastMCP("search-server")

@mcp.tool()
async def web_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """
    Search the web using DuckDuckGo

    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10)

    Returns:
        List of search results with title, url, snippet, and source
    """
    if not query.strip():
        return [{'error': 'Search query cannot be empty'}]

    max_results = max(1, min(max_results, 10))

    try:
        with DDGS() as ddgs:
            results = []
            search_results = ddgs.text(query, max_results=max_results)

            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'source': 'DuckDuckGo'
                })

            return results

    except Exception as e:
        return [{'error': f'Search failed: {str(e)}'}]

@mcp.tool()
async def fetch_webpage(url: str) -> Dict[str, Any]:
    """
    Fetch and extract text content from a webpage

    Args:
        url: The URL to fetch

    Returns:
        Dictionary with title, content, and metadata
    """
    if not url.startswith(('http://', 'https://')):
        return {'error': 'Invalid URL format'}

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return {
                    'title': 'Non-HTML Content',
                    'content': response.text[:5000],
                    'url': str(response.url),
                    'content_type': content_type
                }

            html = response.text

            # Extract title
            title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else 'No Title'

            # Simple text extraction (remove HTML tags)
            text_content = re.sub(r'<[^>]+>', ' ', html)
            text_content = re.sub(r'\s+', ' ', text_content).strip()

            return {
                'title': title,
                'content': text_content[:10000],
                'url': str(response.url),
                'status_code': response.status_code
            }

        except Exception as e:
            return {'error': f'Failed to fetch webpage: {str(e)}'}

@mcp.tool()
async def search_academic_papers(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for academic papers using scholarly search terms

    Args:
        query: Research query
        max_results: Maximum number of papers to return

    Returns:
        List of academic paper results
    """
    scholarly_query = f"{query} academic research paper study"
    results = await web_search(scholarly_query, max_results)

    # Filter and enhance results for academic content
    academic_results = []
    for result in results:
        if 'error' in result:
            academic_results.append(result)
            continue

        url_lower = result.get('url', '').lower()
        snippet_lower = result.get('snippet', '').lower()

        if any(term in url_lower for term in
               ['arxiv', 'scholar.google', 'jstor', 'pubmed', 'ieee', 'acm', 'springer', 'nature']):
            result['type'] = 'academic'
            academic_results.append(result)
        elif any(term in snippet_lower for term in
                ['research', 'study', 'paper', 'journal', 'publication']):
            result['type'] = 'research'
            academic_results.append(result)

    return academic_results

@mcp.resource(uri="mcp://search/v1/info",
              name="search-info",
              description="Information about the search server capabilities")
def get_search_info():
    """Return information about the search server"""
    return {
        'name': 'Search Server',
        'description': 'Provides web search capabilities using DuckDuckGo',
        'tools': [
            'web_search',
            'fetch_webpage',
            'search_academic_papers'
        ],
        'search_engine': 'DuckDuckGo (free, no API key required)'
    }

if __name__ == "__main__":
    mcp.run()