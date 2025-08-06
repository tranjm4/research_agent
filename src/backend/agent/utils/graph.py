"""
File: src/backend/agent/utils/graph.py

This module provides utilities for the agent's graph management
This includes typing definitions
"""

from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict, Annotated
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import ChatMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langgraph.graph.message import add_messages

import json

class ToolNode:
    """
    A node that runs the tools requested in the last AIMessage.
    """
    
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}
        
    def __call__(self, inputs: dict):
        print("Running tools with inputs:", inputs)
        if messages := inputs.get("messages"):
            message = messages[-1]
        else:
            raise ValueError("No messages found in inputs")
        
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                )
            )
        return {"messages": outputs}
    
class State(TypedDict):
    """
    State for the LangGraph agent.
    Contains messages that will be processed by the graph.
    """
    messages: Annotated[list[str], add_messages]
    
    # Intent classification
    intent: Optional[str]
    complexity_indicators: Optional[Dict[str, bool]]
    
    # Planning and verification
    execution_plan: Optional[list[str]] # List of plan steps from planner
    plan_verified: Optional[bool] # Whether the plan has been verified
    plan_feedback: Optional[str] # Feedback form verifier if plan rejected
    plan_status: Optional[str] # "verified", "rejected", "error", "no_task"
    
    # Execution tracking
    execution_status: Optional[str] # "ready", "in_progress", "completed", "failed"
    execution_results: Optional[List[str]] # Results from each execution step
    
    # Metadata for logging purposes
    metadata: Optional[List[Dict[str, str]]] 
    
class PlanState(TypedDict):
    """
    State for the planner.
    Contains the plan and any additional context needed for verification.
    
    Attributes:
        plan (str): The generated plan as a string.
        prompt (str): The prompt used to generate the plan.
        context (List[str]): The previous proposed plans and feedback.
        verified (bool): Whether the plan has been verified.
        feedback (Optional[str]): Feedback from the verifier model, if any.
            - only present if the plan is not verified.
        tools (Optional[List[str]]): Tools available for the planner.
        num_iterations (int): Number of iterations tracked to prevent infinite loops.
    """
    plan: str # The generated plan in the current iteration as a string
    prompt: str # The user prompt or context for the plan
    context: List[str] # Previous proposed plans and feedback
    verified: bool # Whether the plan has been verified
    feedback: Optional[List[str]] # Feedback from the verifier in the current iteration
    tools: Optional[List[str]] # Tools available for the planner
    num_iterations: int # Number of iterations tracked to prevent infinite loops

  
        
class ParallelNode:
    """
    A node that runs multiple Nodes in parallel.
    """
    
    def __init__(self, nodes: list) -> None:
        self.nodes = [(f"node_{i}", node) for i, node in enumerate(nodes)]
        self.graph = self._build_graph()
        
    def aggregator(self, state: State):
        """
        Aggregates the outputs from the parallel nodes into a single state.
        
        Args:
            state (State): The current state of the graph.
        
        Returns:
            State: The updated state with aggregated messages.
        """
        print(state["messages"])
        return {
            "messages": "Hello"
        }
            
    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("aggregator", self.aggregator)
        for node in self.nodes:
            name = node[0]
            node_callable = node[1]
            graph.add_node(name, node_callable)
            graph.add_edge(START, name)
            graph.add_edge(name, "aggregator")
        graph.add_edge("aggregator", END)
        graph = graph.compile(checkpointer=None)
        return graph