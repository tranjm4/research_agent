"""
File: src/backend/agent/graph.py

This file details the graph structure and routing logic
for the LangGraph agent.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Optional, Union, Dict
from typing_extensions import TypedDict


from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain_core.messages import ChatMessage, ToolMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.tools import tool


from utils import ModelWrapper, State
from core_model import CoreModel
from tools.retriever import Retriever
from tools.search import SearchTool

from pathlib import Path
import json

from utils import VERBOSE

class Graph(StateGraph):
    """
    Custom StateGraph class for the LangGraph agent.
    This class extends the StateGraph to include custom routing logic.
    """
    
    def __init__(self, config: Optional[Dict[str, Union[str, int]]] = None):
        super().__init__(State)
        
        self.config = config
        system_prompt = load_file(config["core_config"]["system_prompt"])
        config["core_config"]["system_prompt"] = system_prompt
        
        self.core_model = CoreModel(**config["core_config"])
        

        tools = [
            Retriever(**config["retriever_config"]).as_tool(),
            SearchTool(**config["search_config"]).as_tool()
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

        self.graph = self.compile(checkpointer=MemorySaver())

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
            
            base_path = Path(__file__).parent
            with open(base_path / out_file, "a") as f:
                f.write(json.dumps(metadata) + "\n")
                
            if VERBOSE:
                print()
                print(f"Logged metadata to {out_file}")
                print()
            
            
        return {
            "metadata": {}
        }


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

def __main_loop(graph: StateGraph, verbose=True):
    """
    Main loop to run the LangGraph agent.
    
    Args:
        graph (StateGraph): The graph to run.
    """
    while True:
        try:
            user_input = input(">>> ")
            print()
            if user_input.lower() == 'exit':
                break
            
            if verbose:
                _stream_verbose(graph, user_input)
            else:
                _stream_non_verbose(graph, user_input)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            raise e
        
def _stream_verbose(graph: Graph, user_input: str):
    """
    Streams the output from the graph with verbose logging.
    
    Args:
        graph (Graph): The graph to run.
        user_input (str): The input from the user.
    """
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]},
                                        config={"configurable": {"thread_id": "1"}},
                                        stream_mode="values"):
        for value in event.values():
            print(value[-1].content, flush=True)
            
def _stream_non_verbose(graph, user_input):
    """
    Streams the output from the graph without verbose logging.
    
    Args:
        graph (StateGraph): The graph to run.
        user_input (str): The input from the user.
    """
    for token, metadata in graph.stream({"messages": [{"role": "user", "content": user_input}]},
                                         config={"configurable": {"thread_id": "1"}},
                                         stream_mode="messages"):
        # print(token.content, end="", flush=True)
        try:
            """
            This prints the AI response as it is generated.
            TODO: create a verbose mode that prints all messages for debugging
                when creating a more structured main loop.
            """
            if token.additional_kwargs == {} and token.response_metadata == {}:
                if token.content.startswith("Document") or token.content.startswith("[{"):
                    continue
                print(token.content, end="", flush=True) # this should be the AI response
        except Exception as e:
            continue
        
    print()
    
def load_file(file_path: str) -> str:
    """
    Loads the content of a file.
    
    Args:
        file_path (str): The path to the file to load.
        
    Returns:
        str: The content of the file.
    """
    base_path = Path(__file__).parent
    with open(base_path / "prompts" / file_path, "r") as f:
        return "\n".join([line.strip() for line in f.readlines() if line.strip()])  # Remove empty lines and strip whitespace
