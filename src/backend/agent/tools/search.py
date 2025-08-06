"""
File: agent/tools/search.py

This module defines the Search tool for performing web searches using the DuckDuckGoSearchRun module.   
"""

from langchain_core.tools import tool

from ddgs import DDGS

from dotenv import load_dotenv
load_dotenv()

from utils.logging import log_stats
from utils.typing import ModelConfig

class SearchTool:
    """
    A tool for performing search operations.
    
    It uses the DuckDuckGoSearchRun tool to perform internet searches
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initializes the SearchTool with the DuckDuckGoSearchRun tool.
        """
        params = config.get("params", {})
        
        self.search = DDGS()
        self.num_results = params.get("num_results", 5)
        
        self.metadata = config.get("metadata", {})
        self.logging = config.get("logs", {})
        
        if not self.logging:
            print("No logging configuration provided. Please check your config file.\n")
            raise ValueError(f"No logging configuration provided. Config: {config}")
        
        self.log_stats = log_stats(self.metadata, self.logging["out_file"], self.logging["err_file"])
        self.query = self.log_stats(self._query)
    
    def _query(self, prompt: str) -> list:
        return self.search.text(prompt, max_results=self.num_results)
    
    def as_tool(self):
        """
        Returns the SearchTool as a LangChain tool.
        
        Returns:
            RunnableLambda: A runnable that performs the search operation.
        """
        @tool(name_or_callable="search_tool",
              description="Searches the web for information based on the input query.",
              parse_docstring=True)
        def search_tool(query: str) -> list:
            """Performs a search operation using the DuckDuckGoSearchRun tool.

            This tool uses the DuckDuckGoSearch (ddgs) module to perform searches on the web.

            Args:
                query (str): The search query to call the DuckDuckGoSearchRun tool with.

            Returns:
                list: A list of search results, each containing a title, URL, and snippet.
            """
            return self.query(query)

        return search_tool


if __name__ == "__main__":
    search_tool = SearchTool()
    results = search_tool.invoke()
