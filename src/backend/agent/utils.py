"""
File: src/backend/agent/utils.py

A file consisting of utility classes and functions for the system.
This includes:
- ModelWrapper: A wrapper class for tool models that provides a standardized interface for invoking the model.
- ToolNode: A node that runs the tools requested in the last AIMessage.
- ParallelNode: A node that runs a given list of Nodes in parallel.
- State: A TypedDict that defines the state of the LangGraph agent.

"""

import json

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import BaseMessage

from typing import Annotated, Union, Optional, Dict, List
from typing_extensions import TypedDict

from pathlib import Path
import os

import time
import json

"""
Set the VERBOSE flag to True to enable verbose output.
"""
VERBOSE = False

def log_stats(metadata, out_file, err_file):
    """
    Decorator to log the statistics of the model's call,
    including the number of tokens used and the time taken for the call.
    
    Args:
        metadata (dict): Metadata about the model call, including model type and version name.
        out_file (str): Path to the output file where the logged attributes will be saved.
        err_file (str): Path to the error file where any errors will be logged.
    Returns:
        A decorator that wraps the function to log the statistics.
    """
    def decorator(func):
        
        def wrapper(*args, **kwargs):
            try:
                start_time = time.perf_counter()
                output = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                
                execution_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

                logged_attributes = {
                    "execution_time": execution_time,
                    "execution_date": execution_date,
                    "output": output.__repr__(),
                    "args": args.__repr__(),
                    "kwargs": kwargs.__repr__(),
                }
                for k, v in metadata.items():
                    logged_attributes[k] = v
                    
                # log the attributes to the output file
                out_file_path = Path(__file__).parent / out_file
                if VERBOSE:
                    print(f"Logging attributes to {out_file_path}")
                with open(out_file_path, "a") as f:
                    f.write(json.dumps(logged_attributes) + "\n")

                return output
            except FileNotFoundError as e:
                print(f"Error: {e}. Please check the file paths for out_file and err_file.")
                print(f"Received file paths: out_file={out_file}, err_file={err_file}")
            except Exception as e:
                # log the error to the error file
                err_file_path = Path(__file__).parent / err_file
                with open(err_file_path, "a") as f:
                    error_doc = {
                        "error": str(e),
                        "args": args,
                        "kwargs": kwargs,
                        "metadata": metadata,
                        "execution_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    }
                    f.write(json.dumps(error_doc) + "\n")
                print(f"An error occurred: {e}. Please check the error log at {err_file_path}")
                raise e
            
        return wrapper
    return decorator


class ModelConfig(TypedDict):
    """
    Configuration dictionary for the agent.
    Contains metadata, parameters, and logging information.
    """
    metadata: Dict[str, str]
    params: Dict[str, Union[str, int]]
    logs: Dict[str, str]
    system_prompt: Optional[Path]
    
class GraphConfig(TypedDict):
    """
    Configuration dictionary for the graph.
    Contains core_config, search_config, and retriever_config.
    """
    core_config: ModelConfig
    search_config: ModelConfig
    retriever_config: ModelConfig

class QueryDict(TypedDict):
    """
    A dictionary representing a query with a string input.
    """
    query: str
    result: Union[str, list]

    
class State(TypedDict):
    """
    State for the LangGraph agent.
    Contains messages that will be processed by the graph.
    """
    messages: Annotated[list[str], add_messages]
    
    metadata: Optional[List[Dict[str, str]]]
    
    memory: Optional[Dict[str, Union[str, list]]]
    search_results: Optional[list[QueryDict]]
    db_results: Optional[list[QueryDict]]
    
    tool_log = Optional[List[str]]
    

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
    
class ModelWrapper:
    """
    A wrapper class for LLM models that provides a standardized interface for invoking the model.
    """

    def __init__(self, metadata, logging):
        self.metadata = metadata
        self.logging = logging
        
        
        
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
            The dict should have a key "input" with the input string.

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
        for chunk in self.chain.stream({"input": input_data}):
            yield chunk
        
        
class ParallelNode():
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
    
def call1(state: State):
    """
    A sample function to demonstrate a callable node.
    
    Args:
        state (State): The current state of the graph.
    
    Returns:
        State: The updated state with a message.
    """
    return {"messages": ["1st call"]}

def call2(state: State):
    """
    Another sample function to demonstrate a callable node.
    
    Args:
        state (State): The current state of the graph.
    
    Returns:
        State: The updated state with a message.
    """
    return {"messages": ["2nd call"]}
    
if __name__ == "__main__":
    c1 = RunnableLambda(lambda x: call1(x))
    c2 = RunnableLambda(lambda x: call2(x))
    c3 = RunnableLambda(lambda x: {"messages": ["3rd call"]})
    parallel_node = ParallelNode([c1, c2, c3])

    result = parallel_node.graph.invoke({"input": None, "messages": []})
    
    for r in result["messages"]:
        print(r)