"""
File: src/backend/agent/models/manager_model.py

This file contains the manager agent LLM model, which
is responsible for determining the flow of operations for the langgraph agent.
"""

from langchain.chat_models import init_chat_model
from utils import ModelWrapper
from typing import Optional, Dict, Union, List
from typing_extensions import TypedDict

class ManagerModel(ModelWrapper):
    """
    ManagerModel is a custom model that extends ModelWrapper to manage user documents and provide suggestions.
    It uses a StateGraph to define the flow of the agent's operations.
    """
    
    default_system_prompt = """
    You are a document management assistant.
    Your task is to oversee the user's conversation history and documents.
    You will receive user input and conversation history, and you will suggest actions based on the user's input.
    
    You can suggest the user to add documents to their collection based on the conversation history.
    You can also add notes to documents the user has added to their collection.
    
    You will receive the following tools to interact with the user:
    - suggestions_tool: Suggests actions based on the user's input and conversation history.
    - notes_tool: Adds notes to documents the user has added to their collection.
    """
    
    def __init__(self, config: Optional[Dict[str, Union[str, int]]] = None):
        
        self.model_params = config.get("params", {})
        self.metadata = config.get("metadata", {})
        self.logging = config.get("logs", {})
        
        super().__init__(self.model_params, self.metadata, self.logging)
        
        self.system_prompt = config.get("system_prompt", ManagerModel.default_system_prompt)
        self.tools = None
        self.bound_model = None
        
        
        super().__init__(self.model_params, self.metadata, self.logging)
        
        
    def with_tools(self, tools: List[str]) -> "ManagerModel":
        self.tools = tools
        self.bound_model = self.model.bind_tools(tools)
        return self