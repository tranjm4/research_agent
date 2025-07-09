"""
File: src/agent/tools/wrapper.py

This module provides a wrapper for tool models, allowing them to be invoked with a standardized interface.
"""

from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_ollama import ChatOllama

import json

class ModelWrapper:
    """
    A wrapper class for tool models that provides a standardized interface for invoking the model.
    """

    def __init__(self, agent_type: str, version_name: str, system_prompt: str, input_template: callable, 
                 model: str = "llama3.2:3b", parse_func: callable=None, 
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
        self.agent_type = agent_type
        self.version_name = version_name
        self.model = ChatOllama(model=model, **kwargs)
        self.prompt_template = self.build_prompt_template(system_prompt)
        
        self.parse_func = parse_func if parse_func else lambda x: x  # Default to identity function if no parse_func is provided
        self.parse_func = RunnableLambda(self.parse_func)
        
        self.input_template = RunnableLambda(input_template)
        
        self.chain = self.input_template | self.prompt_template | \
            self.model | RunnableLambda(lambda x: self.log_stats(x)) |self.parse_func
        
    def build_prompt_template(self, system_prompt: str) -> ChatPromptTemplate:
        """
        Builds the prompt template for the tool model.

        Returns:
            ChatPromptTemplate: The constructed prompt template.
        """
        system_message = SystemMessagePromptTemplate.from_template(system_prompt)
        human_message = HumanMessagePromptTemplate.from_template("{input}")

        return ChatPromptTemplate.from_messages([system_message, human_message])
    
    def invoke(self, input_data: dict):
        """
        Invokes the tool model with the provided input data.

        Args:
            input_data (dict): A dictionary containing the input data for the model.

        Returns:
            The output from the model after processing the input.
        """
        # print(f"Invoking {self.agent_type} model with input: {input_data['input']}")
        return self.chain.invoke(input_data)
    
    def stream(self, input_data: dict):
        """
        Streams the output from the tool model based on the provided input data.

        Args:
            input_data (dict): A dictionary containing the input data for the model.

        Returns:
            A generator that yields chunks of output from the model.
        """
        return self.chain.stream(input_data)
    
    def log_stats(self, output):
        """
        Logs the statistics of the model's call,
        including the number of tokens used and the time taken for the call.
        """
        response_metadata = output.response_metadata
        usage_metadata = output.usage_metadata
        logged_attibutes = {
            "model_type": self.agent_type,
            "version_name": self.version_name,
            "model_name": response_metadata["model"],
            "created_at": response_metadata["created_at"],
            "total_duration": response_metadata["total_duration"],
            "load_duration": response_metadata["load_duration"],
            "eval_duration": response_metadata["eval_duration"],
            "prompt_eval_duration": response_metadata["prompt_eval_duration"],
            
            "input_tokens": usage_metadata["input_tokens"],
            "output_tokens": usage_metadata["output_tokens"],
            "total_tokens": usage_metadata["total_tokens"],
        }
        
        # save the logged attributes to a file or database
        json_output = json.dumps(logged_attibutes, indent=4)
        with open("model_stats.jsonl", "a") as f:
            f.write(json_output + "\n")
            
        return output
        