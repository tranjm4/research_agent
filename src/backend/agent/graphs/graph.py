from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from models.model import Model

from typing_extensions import TypedDict
from typing import Annotated, List, Any, Optional, Dict
import contextvars

from utils import logger

# Context variable for MCP session (thread-safe for async)
mcp_session_var = contextvars.ContextVar('mcp_session', default=None)                                                                                                                                                                                                                                                  


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
        Agent node of the graph, wh
        """
        messages = state["messages"]

        if state.get("available_tools"):
            response = await self.model.model.ainvoke(
                messages,
                tools=state["available_tools"]
            )
        else:
            response = await self.model.model.ainvoke(messages)

        return {"messages": [response]}

    async def _tools_node(self, state: State) -> State:
        messages = state["messages"]
        last_message = messages[-1]

        tool_messages = []

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                try:
                    # Get MCP session from context variable
                    mcp_session = mcp_session_var.get()
                    if mcp_session:
                        result = await mcp_session.call_tool(
                            tool_call["name"],
                            arguments=tool_call.get("args", {})
                        )
                    else:
                        result = type('MockResult', (), {'content': f"Tool {tool_call['name']} not available (no session)"})()

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

    async def invoke(self, message: str, conversation_id: str, available_tools=None, mcp_session=None) -> str:
        """

        """
        # Set MCP session in context variable for this async context
        mcp_session_var.set(mcp_session)

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "available_tools": available_tools
        }
        config = {"configurable": {"thread_id": conversation_id}}
        result = await self.graph.ainvoke(initial_state, config=config)

        final_message = result["messages"][-1]
        if hasattr(final_message, 'content'):
            return final_message.content
        else:
            return str(final_message)