"""
File: src/app.py

This script serves as the entry point for the research agent application.

"""
from graphs.core_graph import Graph
from graphs.planner_verifier_graph import PlannerVerifierGraph

from models.core_model import CoreModel
from models.planner_model import PlannerModel
from models.verifier_model import VerifierModel
from tools.retriever import Retriever
from tools.search import SearchTool

from utils.typing import GraphConfig, PlannerVerifierGraphConfig

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from typing import Optional, Dict, Any

from contextlib import asynccontextmanager

import yaml
import json

import sys
import os
from pathlib import Path

import uvicorn
import asyncio

def parse_yaml_config(config_path):
    """Parses the YAML configuration file.
    
    This function reads the configuration file specified by 'config_path', 
    which contains paths to other YAML files.
    
    This function is only run once at the start of the application 
    to load all necessary configurations for the graph's components, 
    such as the core LLM, retriever (and its vector store), and search tool.
    
    Example:
    If the config file contains:
    ```yaml
    core_config: core_config.yaml
    retriever_config: retriever_config.yaml
    search_config: search_config.yaml
    planner_verifier_config: planner_verifier_config.yaml
    ```
    
    Then the function will read each of these files and return a dictionary
    with the configurations for each component.
    
    Example Output:
    ```python
    {
        "core_config": {...},
        "retriever_config": {...},
        "search_config": {...},
        "planner_verifier_config": {...}
    }

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    base_path = Path(__file__).parent / "config"
    config_path = base_path / config_path
    with open(config_path, "r") as f:
        model_config_paths = yaml.safe_load(f)
        
    config = {}
    for key, path in model_config_paths.items():
        path = base_path / path
        
        with open(path, "r") as f:
            config[key] = yaml.safe_load(f)
            
    return config

def init_components(config_path: str) -> Dict[str, Any]:
    """
    Initializes the components of the application based on the provided configuration.
    
    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing initialized components.
    """
    components = {} # Initialize an empty dictionary to hold components
    
    model_configs = parse_yaml_config(config_path) # This should return a dict of configurations
    from pprint import pprint
    pprint(model_configs)
    components["core_model"] = CoreModel(model_configs["core_model"])
    components["retriever"] = Retriever(model_configs["retriever"])
    components["search"] = SearchTool(model_configs["search"])
    components["planner_model"] = PlannerModel(model_configs["planner_model"])
    components["verifier_model"] = VerifierModel(model_configs["verifier_model"])
    components["max_iterations"] = model_configs["planner_verifier"].get("max_iterations", 3)
    
    return components

def init_graph(config_path: str) -> Graph:
    """
    Initializes the graph with the provided configuration.
    
    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Graph: An instance of the Graph class initialized with the configuration.
    """
    components = init_components(config_path)
    
    # Initialize the PlannerVerifierGraph with the components
    # Do this first to be able to initialize the main graph
    planner_verifier_config = PlannerVerifierGraphConfig(
        planner_model=components["planner_model"],
        verifier_model=components["verifier_model"],
        max_iterations=components["max_iterations"]
    )

    planner_verifier_graph = PlannerVerifierGraph(config=planner_verifier_config)
    
    # Initialize the main graph
    main_graph_config = GraphConfig(
        core_model=components["core_model"],
        retriever_tool=components["retriever"],
        search_tool=components["search"],
        planner_verifier_graph=planner_verifier_graph
    )
    main_graph = Graph(config=main_graph_config)

    return main_graph

# CONFIG = parse_yaml_config(os.getenv("GRAPH_CONFIG", "config.yaml"))
CONFIG_PATH = os.getenv("GRAPH_CONFIG", "config.yaml")
GRAPH = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    
    This function initializes the core model when the application starts and cleans it up when the application stops.
    """
    GRAPH["graph"] = init_graph(CONFIG_PATH)
    yield
    
    GRAPH.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust as needed
    allow_credentials=True, # Allows cookies to be included in requests
    allow_methods=["*"],  # Allows all methods, adjust as needed
    allow_headers=["*"],  # Allows all headers, adjust as needed
)

class PromptRequest(BaseModel):
    user_input: str
    conversation_id: str

@app.post("/chat")
async def chat(prompt_request: PromptRequest):
    """
    Endpoint to handle chat requests.
    
    Args:
        prompt_request (PromptRequest): The request body containing user input.
        
    Returns:
        dict: A dictionary containing the response as a stream from the core model.
    """
    user_input = prompt_request.user_input  
    conversation_id = prompt_request.conversation_id
    # print(f"User input: {user_input}")  # Debugging output
    
    async def stream_response():
        for token, metadata in GRAPH["graph"].stream({"messages": [{"role": "user", "content": user_input}]},
                                        config={"configurable": {"thread_id": conversation_id}},
                                        stream_mode="messages"):
            try:
                """
                This prints the AI response as it is generated.
                TODO: create a verbose mode that prints all messages for debugging
                    when creating a more structured main loop.
                """
                if token.additional_kwargs == {} and token.response_metadata == {}:
                    content = str(token.content)
                    if token.content.startswith("Document") or token.content.startswith("[{"):
                        continue
                    
                    chunk = {"type": "token", "content": content}
                    json_line = json.dumps(chunk) + "\n"
                    yield json_line

                    await asyncio.sleep(0.05) # Simulate a delay for streaming effect
            except Exception as e:
                continue
        # Finalize the stream with an end token
        end_chunk = {"type": "end", "content": ""}
        yield json.dumps(end_chunk) + "\n"
    return StreamingResponse(
        stream_response(),
        status_code=200,
        headers={
            "Content-Type": "text/plain; charset=utf-8",
        },
        media_type="text/plain"
    )
    
    # TODO: Log the response and metadata to the database