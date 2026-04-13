import asyncio
import concurrent.futures
import json
import sqlite3
from typing import Annotated, Optional

from ai_prompter import Prompter
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict

from open_notebook.ai.provision import provision_langchain_model
from open_notebook.config import LANGGRAPH_CHECKPOINT_FILE
from open_notebook.domain.notebook import Notebook
from open_notebook.exceptions import OpenNotebookError
from open_notebook.utils import clean_thinking_content
from open_notebook.utils.error_classifier import classify_error
from open_notebook.utils.text_utils import extract_text_content

# Maximum number of tool-calling rounds to prevent infinite loops
MAX_TOOL_ROUNDS = 10


class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]
    notebook: Optional[Notebook]
    context: Optional[str]
    context_config: Optional[dict]
    model_override: Optional[str]
    mcp_tools: Optional[list]


def _run_async_in_new_loop(coro):
    """Run an async coroutine in a new event loop from sync context."""
    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()
        asyncio.set_event_loop(None)


def _run_async(coro):
    """Run an async coroutine, handling both sync and async contexts."""
    try:
        asyncio.get_running_loop()
        # In an event loop — run in a thread with a new loop
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_async_in_new_loop, coro)
            return future.result()
    except RuntimeError:
        # No event loop running
        return asyncio.run(coro)


def _get_mcp_tools():
    """Load MCP tools (LangChain StructuredTools) from configured servers."""
    try:
        from open_notebook.mcp.langchain_bridge import get_mcp_langchain_tools
        tools = _run_async(get_mcp_langchain_tools())
        if tools:
            logger.debug(f"Loaded {len(tools)} MCP tools for chat")
        return tools
    except Exception as e:
        logger.warning(f"Failed to load MCP tools: {e}")
        return []


def call_model_with_messages(state: ThreadState, config: RunnableConfig) -> dict:
    try:
        system_prompt = Prompter(prompt_template="chat/system").render(data=state)  # type: ignore[arg-type]
        payload = [SystemMessage(content=system_prompt)] + state.get("messages", [])
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )

        model = _run_async(
            provision_langchain_model(
                str(payload), model_id, "chat", max_tokens=8192
            )
        )

        # Bind MCP tools to the model if any are available
        mcp_tools = _get_mcp_tools()
        if mcp_tools:
            model = model.bind_tools(mcp_tools)

        ai_message = model.invoke(payload)

        # Gemini returns content as list of dicts — normalize to string
        if hasattr(ai_message, "content") and isinstance(ai_message.content, list):
            text_parts = []
            for part in ai_message.content:
                if isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            if text_parts:
                ai_message = ai_message.model_copy(update={"content": "\n".join(text_parts)})

        # If the model made tool calls, return the raw message (don't clean yet)
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            return {"messages": ai_message, "mcp_tools": mcp_tools}

        # Clean thinking content from AI response (e.g., <think>...</think> tags)
        content = extract_text_content(ai_message.content)
        cleaned_content = clean_thinking_content(content)
        cleaned_message = ai_message.model_copy(update={"content": cleaned_content})

        return {"messages": cleaned_message}
    except OpenNotebookError:
        raise
    except Exception as e:
        error_class, user_message = classify_error(e)
        raise error_class(user_message) from e


def execute_tools(state: ThreadState, config: RunnableConfig) -> dict:
    """Execute tool calls from the model's response."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        return {"messages": []}

    # Build a lookup of available tools — reuse from state if already fetched
    mcp_tools = state.get("mcp_tools") or _get_mcp_tools()
    tool_map = {t.name: t for t in mcp_tools}

    tool_messages = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        tool_call_id = tc.get("id", tool_name)

        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
                # Truncate very long results to avoid blowing up context
                result_str = str(result)
                if len(result_str) > 8000:
                    result_str = result_str[:8000] + "\n... [truncated]"
                tool_messages.append(
                    ToolMessage(content=result_str, tool_call_id=tool_call_id)
                )
            except Exception as e:
                logger.error(f"Tool execution failed for '{tool_name}': {e}")
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: {str(e)}", tool_call_id=tool_call_id
                    )
                )
        else:
            tool_messages.append(
                ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found",
                    tool_call_id=tool_call_id,
                )
            )

    return {"messages": tool_messages}


def should_continue(state: ThreadState) -> str:
    """Decide whether to continue with tools or end."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]

    # If the last message has tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # Safety: count how many tool rounds we've done
        tool_msg_count = sum(1 for m in messages if hasattr(m, 'tool_calls') and m.tool_calls)
        if tool_msg_count >= MAX_TOOL_ROUNDS:
            logger.warning("Max tool rounds reached, stopping")
            return END
        return "tools"

    return END


conn = sqlite3.connect(
    LANGGRAPH_CHECKPOINT_FILE,
    check_same_thread=False,
)
memory = SqliteSaver(conn)

agent_state = StateGraph(ThreadState)
agent_state.add_node("agent", call_model_with_messages)
agent_state.add_node("tools", execute_tools)
agent_state.add_edge(START, "agent")
agent_state.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
agent_state.add_edge("tools", "agent")  # Loop back after tool execution
graph = agent_state.compile(checkpointer=memory)
