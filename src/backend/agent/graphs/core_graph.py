"""
File: src/backend/agent/graph.py

This file details the graph structure and routing logic
for the LangGraph agent.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage
from typing import Optional, Union, Dict, Literal

from graphs.planner_verifier_graph import PlannerVerifierGraph

from dotenv import load_dotenv
import os

# from backend import State
from utils.graph import State
from models.core_model import CoreModel
from tools.retriever import Retriever
from tools.search import SearchTool

from pathlib import Path
import json

from utils.logging import VERBOSE
from utils.typing import GraphConfig

class Graph(StateGraph):
    """
    Custom StateGraph class for the LangGraph agent.
    This class extends the StateGraph to include custom routing logic.
    """

    def __init__(self, config: GraphConfig):
        super().__init__(State)

        self.core_model = config["core_model"]
        self.retriever = config["retriever_tool"]
        self.search = config["search_tool"]
        self.planner_verifier_graph = config["planner_verifier_graph"]

        tools = [
            self.retriever.as_tool(),
            self.search.as_tool(),
        ]
        self.tool_node = ToolNode(tools=tools)
        self.core_model_with_tools = self.core_model.bind_tools(tools)
        
        self.add_node("core_model", self.core_model.node)
        self.add_node("tools", self.tool_node)
        self.add_node("logging", self.log)
        
        self.add_edge(START, "core_model")
        self.add_conditional_edges("core_model", tools_condition)
        self.add_edge("tools", "core_model")
        
        self.add_edge("core_model", "logging")
        self.add_edge("logging", END)
        
        # self.checkpointer = self.__set_postgres_checkpointer() # TODO: Implement PostgresSaver for production
        self.checkpointer = MemorySaver()  # Use MemorySaver for local development

        self.graph = self.compile(checkpointer=self.checkpointer)
        
    def __set_postgres_checkpointer(self):
        """
        Sets up the Postgres checkpointer for the graph. 
        TODO: Implement this for production use.
        
        Returns:
            PostgresSaver: The PostgresSaver instance configured with the database URI.
        """
        load_dotenv()
        db_uri = os.getenv("DB_URI")
        if not db_uri:
            raise ValueError("Database URI is not set in the configuration.")
        
        # return PostgresSaver(db_uri=db_uri, pipe=)

    def stream(self, *args, **kwargs):
        """
        Stream the output from the graph.
        
        Args:
            **kwargs: Additional keyword arguments to pass to the stream method.
            
        Returns:
            Generator yielding the streamed output.
        """
        if VERBOSE:
            for event in self.graph.stream(*args, **kwargs):
                yield event
        else:
            for token, metadata in self.graph.stream(*args, **kwargs):
                yield token, metadata
            
    def invoke(self, *args, **kwargs):
        """
        Invoke the graph with the given arguments.
        
        Args:
            *args: Positional arguments to pass to the invoke method.
            **kwargs: Keyword arguments to pass to the invoke method.
            
        Returns:
            The result of invoking the graph.
        """
        return self.graph.invoke(*args, **kwargs)
    
    def log(self, state: State):
        """
        Logs the metadata and runtime information of the language model's calls
        
        Args:
            state (State): The current state of the graph.
            
        Returns:
            None
        """
        metadata = state.get("metadata", {})
        if not metadata:
            return
        else:
            logging_config = self.config["core_config"]["logs"]
            out_file = logging_config["out_file"]
            
            base_path = Path(__file__).parent.parent
            with open(base_path / out_file, "a") as f:
                f.write(json.dumps(metadata) + "\n")
                
            if VERBOSE:
                print()
                print(f"Logged metadata to {out_file}")
                print()
            
            
        return {
            "metadata": {}
        }
        
    def invoke_planner_verifier(self, state: State) -> Dict[str, any]:
        """
        Invokes the planner-verifier subgraph and adapts the state accordingly.
        
        Args:
            state (State): The current state of the main graph.
        
        Returns:
            Dict[str, any]: The updated state after invoking the planner-verifier graph.
        """
        
        messages = state.get("messages", [])
        if not messages:
            return {"plan_status": "no_task"}
        
        # Extract the user's task from the last message
        task = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        
        # Get available tools as context
        available_tools = [
            "retriever - search the internal knowledge base for research papers",
            "search - search the web for relevant information"
        ]
        
        # plan_state = PlanState(
        #     plan="",
        #     prompt=task,
        #     context=[],
        #     verified=False,
        #     feedback=None,
        #     tools=available_tools
        # )
        
        try:
            result = self.planner_verifier_graph.invoke()
            
            verified = result.get("verified", False)
            plan = result.get("plan", "")
            feedback = result.get("feedback", "")
            
            return {
                "execution_plan": plan,
                "plan_verified": verified,
                "plan_feedback": feedback,
                "plan_status": "verified" if verified else "rejected"
            }
        except Exception as e:
            print(f"Error occurred while invoking planner-verifier: {e}")
            return {
                "excution_plan": [],
                "plan_verified": False,
                "plan_feedback": f"Planning failed: {str(e)}",
                "plan_status": "error"
            }
            
    def execute_plan(self, state: State) -> Dict[str, any]:
        """
        Executes the verified plan step by step and updates the state accordingly.
        
        Args:
            state (State): The current state with execution plan.
        
        Returns:
            Dict: Updated state with execution results.
        """
        execution_results = []
        execution_plan = state.get("execution_plan", [])
        if not execution_plan:
            return {"execution_status": "no_plan", "execution_results": ["No plan to execute"]}
        
        plan_summary = "Executing the following plan:\n"
        for i, step in enumerate(execution_plan):
            plan_summary += f"{i+1}. {step}\n"
            
        messages = state.get("messages", [])
        execution_message = HumanMessage(
            content=f"Please execute this verified plan:\n\n{plan_summary}\n\n Use the available tools as needed."
        )
        
        return {
            "messages": messages + [execution_message],
            "execution_status": "ready"
        }
    
    def route_after_intent(self, state: State) -> Literal["simple", "complex"]:
        """
        Routes based on intent classification.
        
        Args:
            state (State): Current state with intent.
            
        Returns:
            str: Next node to route to based on intent.
        """
        
        return state.get("intent", "simple")
    
    def classify_intent(self, state: State) -> Dict:
        """
        Classifies the user's intent to determine routing strategy.
        
        Args:
            state (State): Current state containing user message.
            
        Returns:
            Dict: Updated state with intent classification.
        """
        messages = state.get("messages", [])
        if not messages:
            return {"intent": "simple"}
        
        last_message = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        
        complex_patterns = [
            "compare", "merge", "analyze across", "summarize_multiple",
            "batch process", "workflow", "extract from multiple",
            "generate report from", "find all", "multi-step",
        ]
        
        # Check for complexity indicators
        # TODO: Improve this logic to be more robust
        is_complex = any(pattern in last_message.lower() for pattern in complex_patterns)
        
        # Check for multi-document indicators
        # TODO: Improve this logic to be more robust
        multi_doc_indicators = ["documents", "files", "all", "multiple", "entire", "batch"]
        has_multi_doc = any(indicator in last_message.lower() for indicator in multi_doc_indicators)
        
        # Check for conditional logic
        # TODO: Improve this logic to be more robust
        conditional_patterns = ["if", "when", "unless", "depending on", "based on"]
        has_conditionals = any(pattern in last_message.lower() for pattern in conditional_patterns)
        
        intent = "complex" if (is_complex or has_multi_doc or has_conditionals) else "simple"
        
        return {
            "intent": intent,
            "complexity_indicators": {
                "is_complex": is_complex,
                "has_multi_doc": has_multi_doc,
                "has_conditionals": has_conditionals
            }
        }
    
    def route_after_planning(self, state: State) -> Literal["execute", "respond"]:
        """
        Routes after planner-verifier subgraph completion.
        
        Args:
            state (State): Current state with planning results.
            
        Returns:
            str: Next node to route to.
        """
        
        plan_status = state.get("plan_status", "rejected")
        
        if plan_status == "verified":
            return "execute"
        elif plan_status == "rejected":
            return "respond"
        elif plan_status == "error":
            return "respond"
        

def route_tools(state: State) -> Optional[str]:
    """
    Routes the tools based on the state of the agent.
    
    Args:
        state (State): The current state of the agent.
        
    Returns:
        Optional[str]: The name of the tool to route to, or None if no tool is needed.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tool_node"
    return END
    
def load_file(file_path: str) -> str:
    """
    Loads the content of a file.
    
    Args:
        file_path (str): The path to the file to load.
        
    Returns:
        str: The content of the file.
    """
    base_path = Path(__file__).parent.parent
    print(base_path)
    with open(base_path / "models" / "prompts" / file_path, "r") as f:
        return "\n".join([line.strip() for line in f.readlines() if line.strip()])  # Remove empty lines and strip whitespace