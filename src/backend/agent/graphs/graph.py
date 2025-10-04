from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, ToolMessage, trim_messages
from langgraph.checkpoint.memory import InMemorySaver

from models.model import Model

from typing_extensions import TypedDict
from typing import Annotated, List, Any, Optional, Dict
import contextvars

from utils import logger

# Configuration for message trimming (in tokens)
# Reserve tokens: 16385 (model limit) - 4096 (max_tokens response) - 1000 (tool definitions) = ~11000 for messages
MAX_MESSAGE_TOKENS = 11000  # Maximum tokens for conversation history

# Context variables for MCP sessions and tool-to-server mapping (thread-safe for async)
mcp_sessions_var = contextvars.ContextVar('mcp_sessions', default=None)
tool_server_map_var = contextvars.ContextVar('tool_server_map', default=None)                                                                                                                                                                                                                                                  


class State(TypedDict):
    messages: Annotated[list, add_messages]
    available_tools: Optional[List[Dict]]


class MCPGraph:
    def __init__(self, model: Model):
        self.model = model
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """
        Graph initialization
        TODO: Implement graph config and parsing framework for flexibility
        """
        graph = StateGraph(State)

        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        graph.add_edge("tools", "agent") # Loops back to the agent for further processing
        memory = InMemorySaver()
        return graph.compile(checkpointer=memory)

    async def _agent_node(self, state: State) -> State:
        """
        Agent node that trims message history to avoid context length issues
        """
        messages = state["messages"]
        logger.info(f"Agent node received {len(messages)} messages from state")

        # Simple approach: keep last 20 messages to avoid context issues
        # This is more reliable than token counting which can fail
        MAX_MESSAGES_TO_SEND = 20
        if len(messages) > MAX_MESSAGES_TO_SEND:
            trimmed_messages = messages[-MAX_MESSAGES_TO_SEND:]
            logger.info(f"Trimmed from {len(messages)} to {len(trimmed_messages)} messages")
        else:
            trimmed_messages = messages

        if state.get("available_tools"):
            response = await self.model.model.ainvoke(
                trimmed_messages,
                tools=state["available_tools"]
            )
        else:
            response = await self.model.model.ainvoke(trimmed_messages)

        return {"messages": [response]}

    async def _tools_node(self, state: State) -> State:
        messages = state["messages"]
        last_message = messages[-1]

        tool_messages = []

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                try:
                    # Get MCP sessions and tool-server mapping from context variables
                    mcp_sessions = mcp_sessions_var.get()
                    tool_server_map = tool_server_map_var.get()

                    if not mcp_sessions or not tool_server_map:
                        result = type('MockResult', (), {'content': f"Tool {tool_call['name']} not available (no sessions)"})()
                    else:
                        # Find which server provides this tool
                        tool_name = tool_call["name"]
                        server_name = tool_server_map.get(tool_name)

                        if not server_name or server_name not in mcp_sessions:
                            result = type('MockResult', (), {'content': f"Tool {tool_name} not found in any server"})()
                        else:
                            # Call the tool on the appropriate server
                            mcp_session = mcp_sessions[server_name]
                            result = await mcp_session.call_tool(
                                tool_name,
                                arguments=tool_call.get("args", {})
                            )

                    tool_message = ToolMessage(
                        content=str(result.content),
                        tool_call_id=tool_call.get("id", "")
                    )
                    tool_messages.append(tool_message)

                except Exception as e:
                    logger.error(f"Error calling tool {tool_call['name']}: {e}")
                    tool_message = ToolMessage(
                        content=f"Error calling tool {tool_call['name']}: {str(e)}",
                        tool_call_id=tool_call.get("id", "")
                    )
                    tool_messages.append(tool_message)

        return {"messages": tool_messages}

    def _should_continue(self, state: State) -> str:
        """
        Checks if the agent node has decided to use a tool
        (if 'tool_calls' exists as an attribute)
        
        Args:
            state (State): The current state of the graph
            
        Returns:
            str: The new node to transition to
        """
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        return "end"

    async def invoke(self, message: str, conversation_id: str, available_tools=None, mcp_sessions=None) -> str:
        """
        Invoke the graph with a message and multiple MCP sessions

        Args:
            message: The user's message
            conversation_id: Thread ID for conversation memory
            available_tools: List of available tools with server metadata
            mcp_sessions: Dict mapping server names to their MCP sessions

        Returns:
            The final response content
        """
        logger.info(f"Processing message for conversation_id: {conversation_id}")

        # Build tool-to-server mapping from available_tools
        tool_server_map = {}
        if available_tools:
            for tool in available_tools:
                tool_name = tool.get("function", {}).get("name")
                server_name = tool.get("server")
                if tool_name and server_name:
                    tool_server_map[tool_name] = server_name

        # Set MCP sessions and mapping in context variables for this async context
        mcp_sessions_var.set(mcp_sessions)
        tool_server_map_var.set(tool_server_map)

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "available_tools": available_tools
        }
        config = {"configurable": {"thread_id": conversation_id}}

        logger.info(f"Invoking graph with thread_id: {conversation_id}")
        result = await self.graph.ainvoke(initial_state, config=config)

        final_message = result["messages"][-1]
        if hasattr(final_message, 'content'):
            return final_message.content
        else:
            return str(final_message)