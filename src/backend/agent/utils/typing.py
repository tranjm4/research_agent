"""
File: agent/utils/typing.py

This module defines custom types and pydantic models for the agent's configuration and state management.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List, Union
from typing_extensions import TypedDict, Annotated
from pathlib import Path


class ModelConfig(TypedDict):
    """
    Configuration dictionary for the agent.
    Contains metadata, parameters, and logging information.
    """
    metadata: Dict[str, str]
    params: Dict[str, Union[str, int]]
    logs: Dict[str, str]
    system_prompt: Optional[Path]
    
class GraphConfig(TypedDict):
    """
    Configuration dictionary for the graph.
    Contains core_config, search_config, and retriever_config.
    """
    core_model: "CoreModel"
    search_tool: "SearchTool"
    retriever_tool: "Retriever"
    planner_verifier_graph: "PlannerVerifierGraph"

class PlannerVerifierGraphConfig(TypedDict):
    """
    Configuration for the planner-verifier graph.
    Contains planner_config and verifier_config.
    """
    planner_model: "PlannerModel"
    verifier_model: "VerifierModel"
    max_iterations: int

class QueryDict(TypedDict):
    """
    A dictionary representing a query with a string input.
    """
    query: str
    result: Union[str, list]
    
class CoreModelConfig(TypedDict):
    """
    Configuration for the core model.
    """
    
    