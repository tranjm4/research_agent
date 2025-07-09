"""
File: agent/tools/search.py

This module defines a search model for the agent, which is used to retrieve relevant information from a vector store.
"""

from agent.tools.wrapper import ModelWrapper
from langchain_core.runnables import RunnableLambda, RunnableSequence
from rag.prompting import keyword_decomposition
from rag.vectorstore.vector_db import load_db

from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

class SearchModel(ModelWrapper):
    """
    A model wrapper for the search functionality, allowing the agent to search through a vector store.
    """

    def __init__(self, system_prompt: str, model_name: str = "llama3.2:3b", **kwargs):
        self.input_template = lambda x: {
            "input": x["input"]
        }
        self.parse_func = lambda x: x

        super().__init__(
            agent_type="search",
            system_prompt=system_prompt,
            input_template=self.input_template,
            model_name=model_name,
            parse_func=self.parse_func,
            **kwargs
        )