"""
File: src/agent/planner.py

This module provides a PlannerModel class that integrates 
with a language model to generate plans based on a user prompt. 
It has search, summary, and task decomposition tools.

Given a system prompt, it constructs a plan of tools to execute 
to help the core model answer a user question.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from agent.tools.wrapper import ModelWrapper

class PlannerModel(ModelWrapper):
  def __init__(self, system_prompt: str, model: str = "llama3.2:1b", **kwargs):
    """
    Initializes the PlannerModel with a system prompt and model name.

    Args:
      system_prompt (str): The system prompt to use for the model.
      model_name (str): The name of the model to use.
      kwargs: Additional keyword arguments for model configuration.
    """
    self.input_template = lambda x: {
      "input": x["input"]
    }
    self.parse_func = lambda x: x
    
    super().__init__(
      agent_type="planner",
      system_prompt=system_prompt,
      input_template=self.input_template,
      model=model,
      parse_func=self.parse_func,  # Default to identity function if no parse_func is provided
      **kwargs
    )
    
  def build_prompt_template(self, system_prompt: str) -> ChatPromptTemplate:
    """
    Builds the prompt template for the planner model.

    Args:
      system_prompt (str): The system prompt to use for the model.

    Returns:
      ChatPromptTemplate: The constructed prompt template.
    """
    return ChatPromptTemplate.from_messages(
      [
        ("system", system_prompt),
        ("human", "{input}")
      ]
    )
    
  def invoke(self, input_data: dict):
    output = super().invoke(input_data)
    print(f"PlannerModel prompt: {input_data['input']}")
    print(f"PlannerModel output: {output.content.strip()}")
    print("-" * 50)
    return output
      
def default_planner_prompt() -> str:
  """
  Returns the default system prompt for the planner model.
  
  Returns:
    str: The default system prompt.
  """
  return """
  INSTRUCTIONS
  You are a planner model that generates a sequence of tools to execute 
  to help the core model answer a user question.
  
  Given a user prompt, construct a plan of tools to execute.
  
  Keep the output concise and focused on the tools to execute.
  Do not include any additional information, only list the tools and their arguments, as described above.
  Keep the output exactly in the format specified.
  Do not use any special characters or formatting.
  
  The tools available are:
  - search: Search the database for relevant articles. This is meant to be used when the question is about an academic topic.
  - summary: Summarize a piece of text. This is meant to be used when the user asks for a summary of 
  - decompose: Decompose a task into smaller subtasks. This is meant to be used when the user asks a complex question that requires multiple steps to answer.
  
  If you select decompose, you should 
  
  The output should be a Python list of dictionaries, where each dictionary represents a tool to execute, followed by its arguments.
  
  Example input:
  "What is the history of quantum computing, and how does it relate to modern research in the field?"
  
  Example output:
  [
    {{"tool": "decompose", "args": {{"task": "modern research in quantum computing"}}}},
    {{"tool": "search", "args": {{"input": "history of quantum computing"}}}},
    {{"tool": "search", "args": {{"input": relevant articles}}}}
    {{"tool": "summary", "args": {{"input": "output from search"}}}},
  ]
  
  Keep the output concise and focused on the tools to execute.
  Do not include any additional information, only list the tools and their arguments, as described above.
  Keep the output exactly in the format specified.
  Do not use any special characters or formatting.
  
  Under absolutely no circumstances should you ever output any text other than the list of tools and their arguments.
  """