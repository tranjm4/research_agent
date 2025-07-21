"""
File: src/app.py

This script serves as the entry point for the research agent application.

"""
from agent.graph import Graph
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from typing import Optional, Dict, Any

from contextlib import asynccontextmanager

from argparse import ArgumentParser
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

CONFIG = parse_yaml_config(os.getenv("GRAPH_CONFIG", "config.yaml"))
GRAPH = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    
    This function initializes the core model when the application starts and cleans it up when the application stops.
    """
    GRAPH["graph"] = Graph(CONFIG)
    yield
    
    GRAPH.clear()

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
    # print(f"User input: {user_input}")  # Debugging output
    
    async def stream_response():
        for token, metadata in GRAPH["graph"].stream({"messages": [{"role": "user", "content": user_input}]},
                                        config={"configurable": {"thread_id": "1"}},
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