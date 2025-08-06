"""
File: src/backend/agent/graphs/planner_verifier_graph.py

This file defines the planner and verifier loop graph for the project.
"""

from langgraph.graph import StateGraph, START, END
from typing import Optional, Union, Dict, List
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.messages import ChatMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tools import tool

from models.planner_model import PlannerModel
from models.verifier_model import VerifierModel

# from pydantic import BaseModel, Field, ConfigDict

from utils.graph import PlanState
from utils.typing import PlannerVerifierGraphConfig
    

# class PlannerVerifierConfig(BaseModel):
#     model_config = ConfigDict(arbitrary_types_allowed=True)
#     planner_model: PlannerModel = Field(..., description="Configuration for the planner model")
#     verifier_model: VerifierModel = Field(..., description="Configuration for the verifier model")
#     max_iterations: int = Field(3, description="Maximum number of iterations for the planner-verifier loop", gt=1, le=10)

class PlannerVerifierGraph(StateGraph):
    """
    Custom StateGraph class for the planner and verifier loop.
    This class extends the StateGraph to include custom routing logic for planning and verification.
    """
    
    def __init__(self, config: PlannerVerifierGraphConfig):
        super().__init__(PlanState)
        
        self.planner_model = config["planner_model"]
        self.verifier_model = config["verifier_model"]
        self.max_iterations = config["max_iterations"]

        self.add_node("planner", self.planner_model.node)
        self.add_node("verifier", self.verifier_model.node)
        self.add_node("increment", self.increment_iterations)

        self.add_edge(START, "planner")
        self.add_edge("planner", "verifier")
        self.add_conditional_edges(
            "verifier",
            self.verifier_conditional,
            {
                "verified": END,
                "not_verified": "increment",
                "max_iterations": END
            }
        )
        self.add_edge("increment", "planner")
        
        self.graph = self.compile()
        
    def verifier_conditional(self, state: PlanState) -> str:
        """
        Determines the next node based on the verification result.
        
        Args:
            state (PlanState): The current state of the planner.
        
        Returns:
            str: The next node to transition to.
        """
        iteration_count = state.get("num_iterations", 0)
        if iteration_count >= self.max_iterations:
            return "max_iterations"
        if state.get("verified", False):
            return "verified"
        else:
            return "not_verified"
        
    
    def increment_iterations(self, state: PlanState) -> PlanState:
        """
        Increments the number of iterations in the state.
        Adds the feedback to the context
        
        Args:
            state (PlanState): The current state of the planner.
        
        Returns:
            PlanState: The updated state with incremented iterations.
        """
        state["num_iterations"] += 1
        if state.get("feedback"):
            state["context"].append(state["feedback"])
        return state
    
    def invoke(self, initial_state: PlanState) -> Dict[str, any]:
        """
        Invokes the planner and verifier graph with the initial state.
        
        Args:
            initial_state (PlanState): The initial state for planning
        
        Returns:
            Dict[str, any]: The final state after executing the graph.
        """
        # Ensure num_iterations is initialized
        if "num_iterations" not in initial_state:
            initial_state["num_iterations"] = 0
        if "context" not in initial_state:
            initial_state["context"] = []
        if "verified" not in initial_state:
            initial_state["verified"] = False
        if "plan" not in initial_state:
            initial_state["plan"] = ""
            
            
        return self.graph.invoke(initial_state)