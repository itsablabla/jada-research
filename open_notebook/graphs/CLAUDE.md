# Graphs Module

LangGraph-based workflow orchestration for content processing, chat interactions, AI-powered transformations, and native workspace tools.

## Key Components

- **`chat.py`**: Conversational agent with message history, notebook context, tool calling (workspace + MCP), and model override support
- **`source_chat.py`**: Source-focused chat with ContextBuilder for insights/content injection and context tracking
- **`ask.py`**: Multi-search strategy agent (generates search terms, retrieves results, synthesizes answers)
- **`source.py`**: Content ingestion pipeline (extract → save → transform with content-core)
- **`transformation.py`**: Single-node transformation executor with prompt templating via ai_prompter
- **`prompt.py`**: Generic pattern chain for arbitrary prompt-based LLM calls
- **`tools.py`**: Minimal tool library (currently just `get_current_timestamp()`)
- **`workspace_tools.py`**: Native LangChain StructuredTools for workspace interaction (create notes, add sources, search)

## Important Patterns

- **Async/sync bridging in graphs**: Both `chat.py` and `source_chat.py` use `asyncio.new_event_loop()` workaround because LangGraph nodes are sync but `provision_langchain_model()` is async
- **State machines via StateGraph**: Each graph compiles to stateful runnable; conditional edges fan out work (ask.py, source.py do parallel transforms)
- **Prompt templating**: `ai_prompter.Prompter` with Jinja2 templates referenced by path ("chat/system", "ask/entry", etc.)
- **Model provisioning via context**: Config dict passed to node via `RunnableConfig`; defaults fall back to state overrides
- **Checkpointing**: `chat.py` and `source_chat.py` use SqliteSaver for message history (LangGraph's built-in persistence)
- **Content extraction**: `source.py` uses content-core library with provider/model from DefaultModels; URLs and files both supported
- **Tool architecture**: Chat graph merges native workspace tools + external MCP tools via `_get_all_tools()`. Both are bound to the model as LangChain StructuredTools. Tool execution loop: `agent → should_continue → tools → agent`
- **Message history sanitization**: `_sanitize_tool_messages()` ensures every AIMessage with `tool_calls` has matching ToolMessages after it. Required for Anthropic's API which rejects orphaned `tool_use` blocks.

## Error Handling in Graphs

All graph nodes use `classify_error()` from `open_notebook.utils.error_classifier` to catch raw LLM provider exceptions and re-raise them as typed `OpenNotebookError` subclasses with user-friendly messages. This ensures that errors from any AI provider (authentication failures, rate limits, model not found, network issues) are surfaced to the user with actionable messages instead of opaque stack traces.

**Pattern in nodes**:
```python
from open_notebook.utils.error_classifier import classify_error

try:
    result = await model.ainvoke(...)
except Exception as e:
    exc_class, message = classify_error(e)
    raise exc_class(message) from e
```

**Tool execution error handling** (`execute_tools` in chat.py):
Tool errors are classified by keyword (timeout, connection, auth, rate limit, validation) and returned as friendly ToolMessage strings instead of raising exceptions. This keeps the chat graph alive — the AI receives the error message and can explain it to the user or retry.

---

## Workspace Tools (`workspace_tools.py`)

### Purpose

Native LangChain StructuredTools that let the chat AI interact with Open Notebook directly — creating notes, adding sources, and searching content. Merged with MCP tools in the chat graph so the AI can seamlessly use both external services and its own workspace.

### Tools Provided

| Tool Name | Args Schema | Description |
|-----------|-------------|-------------|
| `workspace__create_note` | `CreateNoteArgs(title, content, notebook_id)` | Create a note and link it to a notebook. Sets `note_type="ai"` for AI-generated badge. |
| `workspace__add_source_from_url` | `AddSourceFromURLArgs(url, notebook_id)` | Add a URL as a source. Content extraction happens in the background. |
| `workspace__add_source_from_text` | `AddSourceFromTextArgs(title, text, notebook_id)` | Add raw text as a source with fire-and-forget vectorization. |
| `workspace__search` | `SearchWorkspaceArgs(query, max_results=5)` | Semantic vector search across all sources and notes (minimum_score=0.2). |

### Architecture

```
chat.py → _get_all_tools(notebook_id)
              ├── get_workspace_tools(notebook_id)  ← workspace_tools.py
              └── _get_mcp_tools()                  ← mcp/langchain_bridge.py
```

### Async Pattern

Uses the same `_run_async()` / `_run_async_in_new_loop()` pattern as chat.py (ThreadPoolExecutor + new event loop per call). Unlike MCP tools which use a dedicated persistent event loop, workspace tools create and destroy event loops per call — acceptable because they don't maintain session state.

### Error Containment

All tool implementations catch exceptions and return error strings instead of raising. This prevents a single tool failure from crashing the entire chat graph:

```python
def _create_note(title, content, notebook_id):
    try:
        note = _run_async(_do())
        return f"Note created successfully.\n- ID: {note.id}\n..."
    except Exception as e:
        logger.error(f"Workspace tool create_note failed: {e}")
        return f"Error creating note: {e}"
```

### Tool Naming Convention

All workspace tools are prefixed with `workspace__` (double underscore), matching the MCP tool naming convention (`{server_id}__{tool_name}`). This helps the AI and logging distinguish tool sources.

---

## Tool Calling in Chat (`chat.py`)

### Graph Structure

```
START → agent → should_continue → tools → agent → ... → END
                     ↓
                    END (no tool calls or max rounds)
```

### Tool Loading (`_get_all_tools`)

Loads tools from two sources on every model invocation:
1. **Workspace tools**: `get_workspace_tools(notebook_id)` — always available, 4 tools
2. **MCP tools**: `_get_mcp_tools()` — from external servers, cached at client level in `langchain_bridge._client_cache`

### Tool Execution (`execute_tools`)

Builds a `tool_map` from all available tools, then executes each `tool_call` from the AI's response:
- Handles Gemini's JSON-string args (parses if string instead of dict)
- Truncates results > 8000 chars to avoid blowing up context
- Classifies errors by keyword for friendly messages

### Safety: `should_continue`

Counts tool-calling rounds since the last `HumanMessage` only (not full history). Stops at `MAX_TOOL_ROUNDS = 10` to prevent infinite loops.

### Message History Sanitization (`_sanitize_tool_messages`)

Scans message history before sending to the model. For each AIMessage with `tool_calls`, verifies that matching ToolMessages follow. If any are missing (orphaned from interrupted conversations), adds synthetic ToolMessages with `"[Tool call was interrupted and did not return a result]"`. Critical for Anthropic which returns 400 on orphaned `tool_use` blocks.

---

## Quirks & Edge Cases

- **Async loop gymnastics**: ThreadPoolExecutor workaround needed because LangGraph invokes sync nodes but we call async functions; fragile if event loop state changes
- **`clean_thinking_content()` ubiquitous**: Strips `<think>...</think>` tags from model responses (handles extended thinking models)
- **source_chat.py builds context twice**: ContextBuilder runs during node execution to fetch source/insights; rebuilds list from context_data (inefficient but safe)
- **source.py embedding is async**: `source.vectorize()` returns job command ID; not awaited (fire-and-forget)
- **transformation.py nullable source**: Accepts `input_text` or `source.full_text` (falls back to second if first missing)
- **ask.py hard-coded vector_search**: No fallback to text search despite commented code suggesting it was planned
- **SqliteSaver location**: Checkpoints stored in path from `LANGGRAPH_CHECKPOINT_FILE` env var; connection shared across graphs
- **Workspace tools lazy imports**: Tool implementations use deferred `from open_notebook.domain.notebook import ...` to avoid circular imports
- **workspace__add_source_from_url does not auto-extract**: It creates the Source record but content extraction requires the source processing graph to run separately (background job)
- **workspace__add_source_from_text does vectorize**: Unlike URL sources, text sources fire-and-forget `source.vectorize()` immediately
- **Gemini args format**: `execute_tools` handles Gemini returning tool args as JSON strings instead of dicts
- **Orphaned tool_use**: `_sanitize_tool_messages()` adds synthetic ToolMessages for missing tool_results; Anthropic requires strict pairing

## Key Dependencies

- `langgraph`: StateGraph, Send, END, START, SqliteSaver checkpoint persistence
- `langchain_core`: Messages, OutputParser, RunnableConfig, StructuredTool
- `ai_prompter`: Prompter for Jinja2 template rendering
- `content_core`: `extract_content()` for file/URL processing
- `pydantic`: BaseModel, Field for workspace tool arg schemas
- `open_notebook.ai.provision`: `provision_langchain_model()` (async factory with fallback logic)
- `open_notebook.utils.error_classifier`: `classify_error()` for user-friendly LLM error messages
- `open_notebook.domain.notebook`: Domain models (Source, Note, Asset, SourceInsight, vector_search)
- `open_notebook.mcp.langchain_bridge`: `get_mcp_langchain_tools()` for external tool loading
- `loguru`: Logging

## Usage Example

```python
# Invoke a graph with config override
config = {"configurable": {"model_id": "model:custom_id"}}
result = await chat_graph.ainvoke(
    {"messages": [HumanMessage(content="...")], "notebook": notebook},
    config=config
)

# Source processing (content → save → transform)
result = await source_graph.ainvoke({
    "content_state": {...},  # ProcessSourceState from content-core
    "apply_transformations": [t1, t2],
    "source_id": "source:123",
    "embed": True
})

# Get workspace tools for a specific notebook
from open_notebook.graphs.workspace_tools import get_workspace_tools
tools = get_workspace_tools(notebook_id="notebook:abc123")
# Returns 4 StructuredTool instances: create_note, add_source_from_url,
# add_source_from_text, search
```
