"""
File: src/backend/agent/models/planner_model.py

This file defines the planner model for the LangGraph agent.
It receives a task and returns a plan to be executed.
"""

from langchain.chat_models import init_chat_model
from langchain_core.messages import ChatMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from utils.models import ModelWrapper
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict

from utils.graph import PlanState
from utils.typing import ModelConfig

from ast import literal_eval

class PlannerModel(ModelWrapper):
    default_system_prompt = """
    You are a planner model. Your task is to generate a plan based on the provided task and context.
    You are given the following tools to help you:
    {tools}"""
    default_human_prompt = """
    The given task is: {task}
    The previous context is: {context}
    If there is any feedback, it is: {feedback}
    Please generate a plan to accomplish the task.

    The plan should be a list of steps in Python list format (e.g., [Operation1, Operation2, ...]).
    
    You are given the following tools to help you:
    {tools}
    """
    def __init__(self, config: ModelConfig):
        self.model_params = config.get("params", {})
        self.metadata = config.get("metadata", {})
        self.logging = config.get("logging", {})
        super().__init__(self.model_params, self.metadata, self.logging)

        self.system_prompt = config.get("system_prompt", PlannerModel.default_system_prompt)
        self.system_message = SystemMessage(content=self.system_prompt)
        self.model = init_chat_model(**self.model_params)

    def plan(self, task: str, context: str, feedback: str) -> str:
        """
        Generates a plan based on the provided task, context, and feedback.
        
        Args:
            task (str): The task to be accomplished.
            context (str): The context (e.g., previous steps, relevant information).
            feedback (str): Any feedback from previous attempts or steps, given by the verifier model.
            
        Returns:
            str: The generated plan as a python-list of steps.
        """
        # Prepare the prompt for the model
        prompt = PromptTemplate.from_template(PlannerModel.default_human_prompt)
        chain = RunnableSequence(
            RunnableLambda(lambda x: prompt.invoke(x)),
            self.model,
            self._parse_response
        )
        response = chain.invoke({
            "task": task,
            "context": context,
            "feedback": feedback
        })
        
        return response
        
    def _parse_response(self, response: AIMessage) -> List[str]:
        """
        Parses the response from the planner model.
        
        Args:
            response (AIMessage): The response from the planner model.
            
        Returns:
            List[str]: The parsed plan as a list of steps.
        """
        content = response.content.strip()
        try:
            # Attempt to parse the content as a Python list
            parsed_content = literal_eval(content)
            if isinstance(parsed_content, list):
                return parsed_content
            else:
                raise ValueError("The response from the planner model is not a valid Python list.")
        except (SyntaxError, ValueError):
            raise ValueError("The response from the planner model is not a valid Python list.")
        
    def node(self, state: PlanState) -> Dict[str, Any]:
        """
        Node function for the planner model.
        
        Args:
            state (PlanState): The current state of the graph.
            
        Returns:
            Dict[str, Any]: The updated state after planning.
        """
        plan = self.plan(state["prompt"], state["context"], state["feedback"])
        
        # If feedback is provided, we should append it to the context and reset it
        state_updates = {
            "plan": plan,
        }
        if state["feedback"]:
            state_updates["context"] = state["context"] + [f"Feedback: {state['feedback']}"]
            state_updates["feedback"] = ""

        return state_updates
