"""
File: src/backend/agent/utils/models.py

This module provides utility classes for managing models in the agent system.
"""

import time
from typing import Dict, Union
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from utils.graph import State

class ModelWrapper:
    """
    A wrapper class for LLM models that provides a standardized interface for invoking the model.
    """

    def __init__(self, model_params: Dict[str, Union[str, int]], metadata, logging):
        self.model_params = model_params
        self.metadata = metadata
        self.logging = logging
        
        self.model = init_chat_model(**self.model_params)
        self.bound_model = None
        
    def build_prompt_template(self, system_prompt: str) -> ChatPromptTemplate:
        """
        Builds the prompt template for the tool model.

        Returns:
            ChatPromptTemplate: The constructed prompt template.
        """
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template("{input}")

        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def bind_tools(self, tools: list):
        """
        Binds the tools to the model, allowing it to invoke them when needed.
        
        Args:
            tools (list): A list of tool objects that the model can invoke.
        """
        self.bound_model = self.model.bind_tools(tools)
    
    def node(self, state: State):
        """
        Invokes the tool model with the provided input data.

        Args:
            state (State): The current state of the chatbot

        Returns:
            The output from the model after processing the input.
            The metadata is updated with execution time and date.
            The messages are updated with the model's response.
        """
        start_time = time.time()
        execution_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        messages = state["messages"]
        if not messages:
            raise ValueError("No messages found in state")
        
        # Build the chat prompt template
        # Invoke the model with the chat prompt
        result = self.invoke(messages)
        
        # Log the execution time and metadata
        execution_time = time.time() - start_time
        
        metadata = {
            "execution_time": execution_time,
            "execution_date": execution_date,
            **self.metadata
        }
        
        return {
            "messages": [result],
            "metadata": metadata
        }