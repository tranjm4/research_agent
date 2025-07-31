"""
agent/manager_model.py

This file contains the manager agent model, which is responsible for managing user documents
and providing suggestions based on the user's input.
"""

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Optional, Union, Dict, List
from typing_extensions import TypedDict
from langchain_core.messages import ChatMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tools import tool
from utils import ModelWrapper, State

class ManagerState(TypedDict):
    conversation_history: List[ChatMessage]
    user_documents: List[str]
    suggestions: List[str]
    

class ManagerGraph(StateGraph):
    """
    Custom StateGraph to handle supplementary tasks for the user.
    This includes providing suggestions to interact with the interface on behalf of the user.
    Features include:
        - Adding notes to documents the user has added to their collection.
        - Suggesting users to add documents to their collection based on conversation history.
    """
    
    def __init__(self, config: Optional[Dict[str, Union[str, int]]] = None):
        super().__init__()
        self.model = init_chat_model(config)