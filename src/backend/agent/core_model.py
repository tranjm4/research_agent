from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate
from langchain_core.messages import ChatMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel

from agent.tools.rag.prompting import keyword_decomposition as decomposition
from agent.tools.search import SearchTool
from agent.tools.retriever import Retriever

from typing import Dict, Any
from agent.utils import ModelWrapper, State, log_stats

import time

import yaml
import os

from argparse import ArgumentParser

class CoreModel:
    """
    CoreModel class that initializes the core model for the agent.
    It sets up the model, prompt, and runnable sequence for processing input.
    """
    
    default_system_prompt = """
    You are a research expert. Your task is to answer user questions and queries.
    You also have helper models that provide additional context and information,
    such as summarization, search, and decomposition.
    
    You have access to the following tools to give you additional context:
    - search_tool: Searches the web for information based on the input query.
    - retriever_tool: Retrieves relevant documents from a vector database based on the input query.
    
    Your task is to answer the user's prompt based on the context you are given.
    
    If you refer to a document, you must cite the document's url in parentheses. Do not mention the document's number.
    If you use one of these tools and they are not able to find any relevant information, you should say "I don't know" or "I cannot answer that question".

    Answer succinctly and directly, using the provided context and information.
    """

    def __init__(self, **config: Dict[str, Any]):
        """
        Initializes the CoreModel with a prompt template and model name.
        Args:
            config (Dict): Configuration dictionary containing model hyperparameters.
        """
        
        self.metadata = config.get("metadata", {})
        self.model_params = config.get("params", {})
        self.logging = config.get("logs", {})
        
        self.system_prompt = config.get("system_prompt", self.default_system_prompt)
        self.system_message = SystemMessage(content=self.system_prompt)
        
        self.model = init_chat_model(**self.model_params)
        self.bound_model = None
        
        
    def build_chat_prompt(self, messages: list[ChatMessage]):
        """
        Builds the chat prompt template for the core model.
        
        Returns:
            ChatPromptTemplate: The constructed chat prompt template.
        """
        return ChatPromptTemplate.from_messages(
            [
                self.system_message,
                HumanMessagePromptTemplate.from_template(self.default_user_prompt),
                AIMessagePromptTemplate.from_template("{output}"),
            ]
        )
    
    def invoke(self, messages: list[ChatMessage]) -> str:
        """
        Invokes the core model with the provided input data.
        Args:
            messages (list[ChatMessage]): A list of chat messages from the conversation.
        Returns:
            The output from the model after processing the messages.
        """
        if self.bound_model:
            # If the model is bound to tools, use the bound model
            return self.bound_model.invoke(messages)
        else:
            return self.model.invoke(messages)
    
    def stream(self, input_data:dict):
        """
        Streams the output from the core model based on the provided input data.
        Args:
            input_data (dict): A dictionary containing the input data for the model.
        Returns:
            A generator that yields chunks of output from the model.
        """
        return self.chain.stream(input_data)
    
    def cleanup(self):
        """
        Cleans up the resources used by the core model.
        This method can be extended to release any resources or connections held by the model.
        """
        # Placeholder for cleanup logic if needed in the future
        pass
    
    def __call__(self, state: State):
        """
        Calls the core model with the provided state.
        Args:
            state (State): The state object containing the input data and other relevant information.
        Returns:
            The output from the core model after processing the state.
        """
        return {"messages": [self.invoke(state["messages"])]}
    
    def bind_tools(self, tools: list):
        """
        Binds the tools to the core model.
        Args:
            tools (list): A list of tools to bind to the core model.
        Returns:
            The core model with the tools bound.
        """
        
        self.bound_model = self.model.bind_tools(tools)
        return self

    def chatbot(self, state: State):
        """
        Chatbot function that processes the input state and returns the response.
        
        Args:
            state (State): The current state of the chatbot.
            
        Returns:
            dict: A dictionary containing the response from the core model.
        """
        execution_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        start_time = time.time()
        result = [self.invoke(state["messages"])]
        execution_time = time.time() - start_time
        
        return {
            "messages": result,
            "metadata": {
                "execution_date": execution_date,
                "execution_time": execution_time,
                "output": result.__repr__(),
                "model_name": self.metadata.get("model_name"),
                "type": self.metadata.get("type", "core_model"),
                "args": state["messages"].__repr__(),
                "version_name": self.metadata.get("version_name")
            }
        }
