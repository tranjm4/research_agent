"""
File: src/agent/tools/summary.py

This module provides a model wrapper for summarizing text using a language model.
"""

from agent.tools.wrapper import ToolModelWrapper

class SummaryModel(ToolModelWrapper):
    def __init__(self, system_prompt, model, **kwargs):
        self.input_template = lambda x: {
            "input": x["input"]
        }
        self.parse_func = lambda x: x
        
        super().__init__(
            agent_type="summary",
            system_prompt=system_prompt,
            input_template=self.input_template,
            model_name=model,
            parse_func=self.parse_func,
            **kwargs
        )
        
    
    
def get_system_prompt():
    """
    Base system prompt for the summary model (for prototyping purposes).
    """
    
    prompt = """
    You are a summarization expert in various academic domains. Your task is to summarize the provided text
    in a concise and informative manner, capturing the key points and insights.
    
    """
    
    return prompt