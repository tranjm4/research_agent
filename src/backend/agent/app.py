"""
File: src/backend/agent/host/app.py

This file contains the host application for the system,
where the core language model logic exists.

It interfaces with the core server (Go)
i.e., Frontend <-> Go <-> [host/app.py] <-> MCP servers

It also interfaces with the MCP servers, which provide necessary context
and tool availability
"""
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from models.model import Model, ModelConfig
from graphs.graph import MCPGraph

from contextlib import AsyncExitStack, asynccontextmanager
import asyncio

from typing import Optional, AsyncGenerator
import os
import yaml
import json

from utils import logger

from argparse import ArgumentParser

from dotenv import load_dotenv
load_dotenv()


def load_model_config(config_path: str) -> dict:
    """Load model configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    

# Global MCP client instance
mcp_client = None
mcp_tools = []
mcp_resources = []

# Runs after if __name__ == '__main__' block
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup - test MCP server availability
    if mcp_client:
        try:
            tools = await mcp_client.get_available_tools()
            logger.info(f"MCP server available with tools: {[tool['function']['name'] for tool in tools]}")
            
            # Cache tools
            global mcp_tools
            mcp_tools = tools
        except Exception as e:
            logger.info(f"MCP server check failed: {e}")

    yield

    # Shutdown - nothing to cleanup (stateless)


class MCPClient:
    def __init__(self, model_config: ModelConfig):
        self.server_path = os.getenv("MCP_SERVER_PATH", "./servers/search_server.py")

        if model_config is None:
            model_config = ModelConfig(config=OpenAIConfig(
                model_provider="openai",
                max_tokens=4096,
                temperature=0.7,
                timeout=30,
                max_retries=3
            ))

        self.model = Model(model_config)
        self.graph = MCPGraph(self.model)

    async def get_available_tools(self) -> list:
        """Get available tools from MCP server (stateless)"""
        command = "python"
        server_params = StdioServerParameters(
            command=command,
            args=[self.server_path],
            env=None
        )
        
        # Return cached result if it already exists
        global mcp_tools
        if mcp_tools:
            return mcp_tools
        
        # Otherwise, we make a request to the server for its tools
        async with AsyncExitStack() as exit_stack:
            # Create temporary connection to get tools
            stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await exit_stack.enter_async_context(ClientSession(stdio, write))

            await session.initialize()

            try:
                response = await session.list_tools()
                return [{
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                } for tool in response.tools]
            except Exception as e:
                logger.info(f"Error getting tools: {e}")
                return []

    async def process_query(self, query: str, conversation_id: str) -> str:
        """
        Process query using stateless MCP connection (follows LangGraph patterns)
        """
        try:
            command = "python"
            server_params = StdioServerParameters(
                command=command,
                args=[self.server_path],
                env=None
            )

            async with AsyncExitStack() as exit_stack:
                # Create fresh MCP connection per request (stateless)
                stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
                stdio, write = stdio_transport
                session = await exit_stack.enter_async_context(ClientSession(stdio, write))

                await session.initialize()

                available_tools = mcp_tools

                # Process query with fresh session
                result = await self.graph.invoke(
                    message=query,
                    conversation_id=conversation_id,
                    available_tools=available_tools,
                    mcp_session=session
                )

                # Session automatically cleaned up when exiting context
                return result

        except Exception as e:
            import traceback
            full_error = traceback.format_exc()
            logger.error(f"Error in process_query: {full_error}")
            raise
        
        
app = FastAPI(lifespan=lifespan)


async def stream_response(message: str, conversation_id: str) -> AsyncGenerator:
    """Generator function to stream response tokens
    
    Args:
        message (str): The user input prompt
        conversation_id (str): The conversation ID to distinguish as thread_id
            for the graph
    
    Returns:
        AsyncGenerator: The response stream as an AsyncGenerator
    """
    try:
        logger.info(f"Streaming response for message: {message}")

        # Process the query and get streaming response
        response = await mcp_client.process_query(message, conversation_id)

        # For now, simulate streaming by breaking response into chunks
        words = response.split()
        for word in enumerate(words):
            chunk = {
                "id": 0,
                "content": word[1] + " ",
                "timestamp": datetime.now().isoformat(),
                "sender": "assistant"
            }
            yield f"{json.dumps(chunk)}\n"
            await asyncio.sleep(0.05)  # Small delay to simulate streaming

    except Exception as e:
        import traceback
        full_error = traceback.format_exc()
        logger.error(f"Error in stream_response: {full_error}")

        error_chunk = {
            "id": 0,
            "content": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat(),
            "sender": "error"
        }
        yield f"{json.dumps(error_chunk)}\n"

class ChatPostBody(BaseModel):
    message: str
    conversation_id: str

@app.post("/chat")
async def chat_endpoint(body: ChatPostBody):
    """
    Handle chat requests from the frontend with streaming response
    """
    logger.info(f"Received chat POST request: {body}")

    return StreamingResponse(
        stream_response(body.message, body.conversation_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "mcp_connected": mcp_client.session is not None
    }

def create_app_with_config(config_path: str):
    """Create the app with the loaded configuration"""
    global mcp_client

    # Load model configuration
    config_data = load_model_config(config_path)

    # Restructure the config to match the new Pydantic model structure
    model_params = config_data.get('model_params', {})
    model_params["model_provider"] = config_data.get('model_provider')
    model_config = ModelConfig(params=model_params)
    mcp_client = MCPClient(model_config)

    return app

if __name__ == '__main__':
    parser = ArgumentParser(description="Run FastAPI application with custom arguments")
    # TODO: move --host and --port to .env file
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--config", required=True, help="Config file to read from host/models/config")

    args = parser.parse_args()

    # Create app with configuration
    configured_app = create_app_with_config(args.config)

    uvicorn.run(configured_app, host=args.host, port=args.port)