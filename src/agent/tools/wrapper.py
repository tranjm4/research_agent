"""
File: src/agent/tools/wrapper.py

This module provides a wrapper for tool models, allowing them to be invoked with a standardized interface.
"""

from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePrompt
from langchain_ollama import ChatOllama

class ToolModelWrapper:
    """
    A wrapper class for tool models that provides a standardized interface for invoking the model.
    """

    def __init__(self, system_prompt: str, input_template: callable, 
                 model_name: str = "llama3.2:3b", parse_func: callable=None, 
                 **kwargs):
        """
        Initializes the ToolModelWrapper with a system prompt and model name.

        Args:
            system_prompt (str): The system prompt to use for the model.
            input_template (callable): A lambda or function that formats the input data for the model.
            parse_func (callable, optional): A function to parse the model's output. Defaults to None.
            model_name (str): The name of the model to use.
            kwargs: Additional keyword arguments for model configuration.
        """
        self.model = ChatOllama(model_name=model_name, **kwargs)
        self.prompt_template = self.build_prompt_template(system_prompt)
        
        self.parse_func = parse_func if parse_func else lambda x: x  # Default to identity function if no parse_func is provided
        self.parse_func = RunnableLambda(self.parse_func)
        
        self.input_template = RunnableLambda(input_template)
        
        self.chain = self.input_template | self.prompt_template | self.model | self.parse_func
        
    def build_prompt_template(self, system_prompt: str) -> ChatPromptTemplate:
        """
        Builds the prompt template for the tool model.

        Returns:
            ChatPromptTemplate: The constructed prompt template.
        """
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePrompt.from_template("{input}")

        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def invoke(self, input_data: dict):
        """
        Invokes the tool model with the provided input data.

        Args:
            input_data (dict): A dictionary containing the input data for the model.

        Returns:
            The output from the model after processing the input.
        """
        return self.chain.invoke(input_data)