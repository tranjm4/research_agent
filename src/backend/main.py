"""
File: src/main.py

This script serves as the entry point for the research agent application.

"""
from agent.utils import VERBOSE
from agent.core_model import CoreModel
from agent.graph import Graph, _stream_non_verbose, _stream_verbose, build_graph
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from argparse import ArgumentParser

from pathlib import Path
import yaml

def parse_args():
    """
    Parses command line arguments for the application.
    
    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = ArgumentParser(description="Research Agent Application")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    args = parser.parse_args()
    config_path = args.config
    return parse_yaml_config(config_path)

def parse_yaml_config(config_path):
    """
    Parses the YAML configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    base_path = Path(__file__).parent / "agent" /"config"
    config_path = base_path / config_path
    with open(config_path, "r") as f:
        model_config_paths = yaml.safe_load(f)
        
    config = {}
    for key, path in model_config_paths.items():
        path = base_path / path
        
        with open(path, "r") as f:
            config[key] = yaml.safe_load(f)
            
    return config

CONFIG = parse_args()
GRAPH = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    
    This function initializes the core model when the application starts and cleans it up when the application stops.
    """
    GRAPH["graph"] = Graph(CONFIG)
    await GRAPH["graph"]
    yield
    
    await GRAPH.clear()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, adjust as needed
    allow_headers=["*"],  # Allows all headers, adjust as needed
)

class PromptRequest(BaseModel):
    user_input: str
    
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
    response = await CoreModel.invoke({"input": user_input})
    return {"response": response}


if __name__ == "__main__":
    graph = Graph(CONFIG)  
    while True:
        user_input = input(">>> ")
        if user_input.lower() == "exit":
            break
        
        if VERBOSE:
            _stream_verbose(graph, user_input)
        else:
            _stream_non_verbose(graph, user_input)
