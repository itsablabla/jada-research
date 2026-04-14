import asyncio
import concurrent.futures
import json
import re
import sqlite3
import time
from typing import Annotated, List, Optional

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

# Tool cache to avoid reloading MCP tools on every graph node transition
_tool_cache: dict = {"tools": None, "notebook_id": None, "timestamp": 0.0}
_TOOL_CACHE_TTL = 120  # seconds

# Maximum MCP tools to bind to the model per call (keeps prompt manageable)
MAX_MCP_TOOLS_FOR_MODEL = 40

# Words to ignore when scoring tool relevance
_STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "not", "no", "if", "then",
    "than", "that", "this", "these", "those", "it", "its", "my", "your",
    "our", "their", "his", "her", "me", "him", "them", "us", "we", "you",
    "i", "what", "which", "who", "whom", "how", "when", "where", "why",
    "all", "each", "every", "any", "some", "most", "other", "more",
    "about", "up", "out", "so", "just", "also", "very", "as",
    "using", "use", "find", "get", "make", "please", "want", "need",
})


def _sanitize_tool_messages(messages: list) -> list:
    """Ensure every AIMessage with tool_calls has matching ToolMessages after it.

    Anthropic's API requires that every tool_use block is immediately followed
    by a tool_result block. Orphaned tool_use messages (from interrupted
    conversations, timeouts, or errors) will cause a 400 error.

    This function scans the message history and adds synthetic ToolMessages
    for any tool_calls that lack a corresponding tool_result.
    """
    if not messages:
        return messages

    sanitized = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        sanitized.append(msg)

        # Check if this is an AIMessage with tool_calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call_ids = {tc.get("id") or tc.get("name", f"tc_{j}") for j, tc in enumerate(msg.tool_calls)}

            # Collect tool_result IDs from immediately following ToolMessages
            found_ids = set()
            j = i + 1
            while j < len(messages) and isinstance(messages[j], ToolMessage):
                found_ids.add(getattr(messages[j], "tool_call_id", None))
                j += 1

            # Add synthetic ToolMessages for any missing tool_call_ids
            missing_ids = tool_call_ids - found_ids
            for missing_id in missing_ids:
                logger.warning(
                    f"Adding synthetic tool_result for orphaned tool_call: {missing_id}"
                )
                sanitized.append(
                    ToolMessage(
                        content="[Tool call was interrupted and did not return a result]",
                        tool_call_id=missing_id,
                    )
                )

        i += 1

    return sanitized


class ThreadState(TypedDict):
    messages: Annotated[list, add_messages]
    notebook: Optional[Notebook]
    context: Optional[str]
    context_config: Optional[dict]
    model_override: Optional[str]


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


def _extract_keywords(text: str) -> set:
    """Extract meaningful keywords from text, ignoring stop words."""
    words = set(re.findall(r'\b[a-z]{2,}\b', text.lower()))
    return words - _STOP_WORDS


def _get_last_human_message(messages: list) -> str:
    """Get the text content of the last human message."""
    for msg in reversed(messages):
        if getattr(msg, 'type', None) == 'human':
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            return content
    return ""


def _filter_tools_for_model(all_tools: list, message_hint: str) -> list:
    """Filter tools by relevance to the user's message.

    Always includes all workspace tools. Scores MCP tools by keyword
    overlap with tool name and description, returns top N most relevant.
    This keeps the LLM prompt manageable (~40 tools vs 189).
    """
    ws_tools = [t for t in all_tools if t.name.startswith("workspace__")]
    mcp_tools = [t for t in all_tools if not t.name.startswith("workspace__")]

    if len(mcp_tools) <= MAX_MCP_TOOLS_FOR_MODEL:
        return all_tools  # Few enough already

    if not message_hint:
        return ws_tools + mcp_tools[:MAX_MCP_TOOLS_FOR_MODEL]

    keywords = _extract_keywords(message_hint)
    if not keywords:
        return ws_tools + mcp_tools[:MAX_MCP_TOOLS_FOR_MODEL]

    scored: List[tuple] = []
    for tool in mcp_tools:
        name_lower = tool.name.lower().replace("__", " ").replace("_", " ")
        desc_lower = (tool.description or "").lower()
        score = 0
        for kw in keywords:
            if kw in name_lower:
                score += 5  # Name matches are strong signals
            if kw in desc_lower:
                score += 1
        scored.append((score, tool))

    scored.sort(key=lambda x: -x[0])
    filtered = [t for _, t in scored[:MAX_MCP_TOOLS_FOR_MODEL]]

    logger.info(
        f"Filtered {len(mcp_tools)} MCP tools to {len(filtered)} "
        f"(keywords: {', '.join(sorted(list(keywords)[:10]))})"
    )
    return ws_tools + filtered


def _load_all_tools(notebook_id: Optional[str] = None):
    """Load all tools: native workspace tools + MCP tools.

    Results are cached for _TOOL_CACHE_TTL seconds to avoid reconnecting
    to MCP servers on every graph node transition.
    """
    global _tool_cache

    now = time.time()
    if (
        _tool_cache["tools"] is not None
        and _tool_cache["notebook_id"] == notebook_id
        and (now - _tool_cache["timestamp"]) < _TOOL_CACHE_TTL
    ):
        logger.debug(f"Using cached tools ({len(_tool_cache['tools'])} total)")
        return _tool_cache["tools"]

    tools = []

    # 1. Native workspace tools (always available)
    try:
        from open_notebook.graphs.workspace_tools import get_workspace_tools
        ws_tools = get_workspace_tools(notebook_id=notebook_id)
        tools.extend(ws_tools)
        logger.info(f"Loaded {len(ws_tools)} workspace tools (notebook_id={notebook_id})")
    except Exception as e:
        logger.warning(f"Failed to load workspace tools: {e}")

    # 2. MCP tools from external servers
    mcp_tools = _get_mcp_tools()
    tools.extend(mcp_tools)

    logger.info(f"Total tools available: {len(tools)} ({len(tools) - len(mcp_tools)} workspace + {len(mcp_tools)} MCP)")

    _tool_cache = {"tools": tools, "notebook_id": notebook_id, "timestamp": now}
    return tools


def call_model_with_messages(state: ThreadState, config: RunnableConfig) -> dict:
    try:
        system_prompt = Prompter(prompt_template="chat/system").render(data=state)  # type: ignore[arg-type]
        raw_messages = state.get("messages", [])
        # Sanitize message history to fix orphaned tool_use without tool_result
        # (Anthropic rejects messages where tool_use lacks matching tool_result)
        sanitized_messages = _sanitize_tool_messages(raw_messages)
        payload = [SystemMessage(content=system_prompt)] + sanitized_messages
        model_id = config.get("configurable", {}).get("model_id") or state.get(
            "model_override"
        )

        model = _run_async(
            provision_langchain_model(
                str(payload), model_id, "chat", max_tokens=8192
            )
        )

        # Fetch tools: load all (cached), then filter by relevance to message
        notebook = state.get("notebook")
        notebook_id = notebook.id if notebook and hasattr(notebook, "id") else None
        all_tools = _load_all_tools(notebook_id=notebook_id)
        message_hint = _get_last_human_message(raw_messages)
        filtered_tools = _filter_tools_for_model(all_tools, message_hint)
        if filtered_tools:
            model = model.bind_tools(filtered_tools)

        ai_message = model.invoke(payload)

        # Normalize content format (e.g. Gemini returns list of parts)
        raw_content = ai_message.content
        normalized_content = extract_text_content(raw_content)

        # If the model made tool calls, normalize content and cache tools in state
        if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
            if normalized_content != raw_content:
                ai_message = ai_message.model_copy(update={"content": normalized_content})
            return {"messages": ai_message}

        # Clean thinking content from AI response (e.g., <think>...</think> tags)
        cleaned_content = clean_thinking_content(normalized_content)
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

    # Build a lookup from the FULL tool set (not filtered) so any tool the
    # model was bound with can be executed even across filter boundaries.
    notebook = state.get("notebook")
    notebook_id = notebook.id if notebook and hasattr(notebook, "id") else None
    all_tools = _load_all_tools(notebook_id=notebook_id)
    tool_map = {t.name: t for t in all_tools}

    tool_messages = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc.get("args", {})
        # Gemini may return args as a JSON string instead of dict
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
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
                error_str = str(e).lower()
                # Classify MCP errors for user-friendly messages
                if "timeout" in error_str or "timed out" in error_str:
                    friendly = f"The tool '{tool_name}' timed out. The external service may be slow or unavailable. Try again shortly."
                elif "connection" in error_str or "connect" in error_str:
                    friendly = f"Could not connect to the service for '{tool_name}'. The server may be down."
                elif "401" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
                    friendly = f"Authentication failed for '{tool_name}'. The API credentials may be invalid or expired."
                elif "rate limit" in error_str or "429" in error_str:
                    friendly = f"Rate limit reached for '{tool_name}'. Please wait a moment before trying again."
                elif "validation" in error_str or "invalid" in error_str:
                    friendly = f"The tool '{tool_name}' received invalid input: {e}"
                else:
                    friendly = f"The tool '{tool_name}' encountered an error: {e}"
                logger.error(f"Tool execution failed for '{tool_name}': {e}")
                tool_messages.append(
                    ToolMessage(
                        content=friendly, tool_call_id=tool_call_id
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
        # Safety: count tool rounds since last human message only
        last_human_idx = max(
            (i for i, m in enumerate(messages) if getattr(m, 'type', None) == 'human'),
            default=0,
        )
        current_exchange = messages[last_human_idx:]
        tool_msg_count = sum(1 for m in current_exchange if hasattr(m, 'tool_calls') and m.tool_calls)
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
