"""
Native workspace tools for the chat AI.

These LangChain StructuredTools let the chat agent interact with the
Open Notebook workspace directly — creating notes, adding URL sources,
and searching existing content.  They are merged with MCP tools in the
chat graph so the AI can seamlessly use both external services and
its own workspace.
"""

import asyncio
import concurrent.futures
from typing import List, Optional

from langchain_core.tools import StructuredTool
from loguru import logger
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Async helper (same pattern as chat.py)
# ---------------------------------------------------------------------------

def _run_async_in_new_loop(coro):
    new_loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(new_loop)
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()
        asyncio.set_event_loop(None)


def _run_async(coro):
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_async_in_new_loop, coro)
            return future.result()
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tool argument schemas
# ---------------------------------------------------------------------------

class CreateNoteArgs(BaseModel):
    title: str = Field(description="A short descriptive title for the note")
    content: str = Field(description="The full markdown content of the note")
    notebook_id: str = Field(
        description="The notebook ID to save the note to (e.g. 'notebook:abc123')"
    )


class AddSourceFromURLArgs(BaseModel):
    url: str = Field(description="The URL to ingest as a new source")
    notebook_id: str = Field(
        description="The notebook ID to add the source to (e.g. 'notebook:abc123')"
    )


class AddSourceFromTextArgs(BaseModel):
    title: str = Field(description="Title for the text source")
    text: str = Field(description="The full text content to save as a source")
    notebook_id: str = Field(
        description="The notebook ID to add the source to (e.g. 'notebook:abc123')"
    )


class SearchWorkspaceArgs(BaseModel):
    query: str = Field(description="The search query to find relevant sources and notes")
    max_results: int = Field(
        default=5, description="Maximum number of results to return"
    )


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _create_note(title: str, content: str, notebook_id: str) -> str:
    """Create a note in the workspace and link it to a notebook."""
    logger.info(f"Workspace tool create_note called: title='{title}', notebook_id='{notebook_id}'")
    try:
        from open_notebook.domain.notebook import Note

        async def _do():
            note = Note(title=title, content=content, note_type="ai")
            await note.save()
            logger.info(f"Note saved with id={note.id}")
            if notebook_id:
                await note.add_to_notebook(notebook_id)
                logger.info(f"Note {note.id} linked to notebook {notebook_id}")
            return note

        note = _run_async(_do())
        return (
            f"Note created successfully.\n"
            f"- ID: {note.id}\n"
            f"- Title: {note.title}\n"
            f"- Notebook: {notebook_id}"
        )
    except Exception as e:
        logger.error(f"Workspace tool create_note failed: {e}", exc_info=True)
        return f"Error creating note: {e}"


def _add_source_from_url(url: str, notebook_id: str) -> str:
    """Add a URL as a new source and trigger content extraction."""
    logger.info(f"Workspace tool add_source_from_url called: url='{url}', notebook_id='{notebook_id}'")
    try:
        from open_notebook.domain.notebook import Asset, Source

        async def _do():
            source = Source(
                title=url,
                asset=Asset(url=url),
            )
            await source.save()
            logger.info(f"Source saved with id={source.id}")
            if notebook_id:
                await source.add_to_notebook(notebook_id)
                logger.info(f"Source {source.id} linked to notebook {notebook_id}")
            return source

        source = _run_async(_do())
        return (
            f"Source created from URL. Content extraction will happen in the background.\n"
            f"- ID: {source.id}\n"
            f"- URL: {url}\n"
            f"- Notebook: {notebook_id}\n"
            f"Note: The source text will be available after processing completes."
        )
    except Exception as e:
        logger.error(f"Workspace tool add_source_from_url failed: {e}", exc_info=True)
        return f"Error adding source from URL: {e}"


def _add_source_from_text(title: str, text: str, notebook_id: str) -> str:
    """Add raw text as a new source."""
    logger.info(f"Workspace tool add_source_from_text called: title='{title}', notebook_id='{notebook_id}'")
    try:
        from open_notebook.domain.notebook import Source

        async def _do():
            source = Source(title=title, full_text=text)
            await source.save()
            logger.info(f"Source saved with id={source.id}")
            if notebook_id:
                await source.add_to_notebook(notebook_id)
                logger.info(f"Source {source.id} linked to notebook {notebook_id}")
            # Fire-and-forget vectorization
            try:
                await source.vectorize()
            except Exception as vec_err:
                logger.warning(f"Vectorization failed for {source.id}: {vec_err}")
            return source

        source = _run_async(_do())
        return (
            f"Text source created and queued for embedding.\n"
            f"- ID: {source.id}\n"
            f"- Title: {title}\n"
            f"- Notebook: {notebook_id}"
        )
    except Exception as e:
        logger.error(f"Workspace tool add_source_from_text failed: {e}", exc_info=True)
        return f"Error adding text source: {e}"


def _search_workspace(query: str, max_results: int = 5) -> str:
    """Search across all sources and notes using semantic vector search."""
    try:
        from open_notebook.domain.notebook import vector_search

        async def _do():
            results = await vector_search(
                keyword=query,
                results=max_results,
                source=True,
                note=True,
                minimum_score=0.2,
            )
            return results

        results = _run_async(_do())

        if not results:
            return "No results found for your query."

        lines = [f"Found {len(results)} result(s) for '{query}':\n"]
        for r in results:
            rid = r.get("id", "unknown")
            title = r.get("title") or r.get("name") or "Untitled"
            score = r.get("score", 0)
            content_preview = r.get("content", r.get("full_text", ""))
            if content_preview and len(content_preview) > 200:
                content_preview = content_preview[:200] + "..."
            lines.append(
                f"- [{rid}] {title} (relevance: {score:.2f})\n"
                f"  {content_preview}"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Workspace tool search_workspace failed: {e}")
        return f"Error searching workspace: {e}"


# ---------------------------------------------------------------------------
# Public API — returns the list of workspace StructuredTools
# ---------------------------------------------------------------------------

def get_workspace_tools(notebook_id: Optional[str] = None) -> List[StructuredTool]:
    """
    Build the list of native workspace tools.

    If *notebook_id* is provided, the create/add tools will default to that
    notebook — but the AI can still override via the tool argument.
    """
    tools: List[StructuredTool] = []

    tools.append(
        StructuredTool(
            name="workspace__create_note",
            description=(
                "[Workspace] Create a new note in the current notebook. "
                "Use this to save summaries, observations, extracted data, "
                "or any other text the user wants to keep as a note."
            ),
            func=_create_note,
            args_schema=CreateNoteArgs,
        )
    )

    tools.append(
        StructuredTool(
            name="workspace__add_source_from_url",
            description=(
                "[Workspace] Add a URL as a new source in the current notebook. "
                "The URL content will be extracted and indexed automatically. "
                "Use this when the user wants to save a web page, article, or "
                "online document as a research source."
            ),
            func=_add_source_from_url,
            args_schema=AddSourceFromURLArgs,
        )
    )

    tools.append(
        StructuredTool(
            name="workspace__add_source_from_text",
            description=(
                "[Workspace] Add raw text as a new source in the current notebook. "
                "Use this when the user provides or you generate text content "
                "(e.g. email bodies, pasted text, generated reports) that should "
                "be saved as a searchable source."
            ),
            func=_add_source_from_text,
            args_schema=AddSourceFromTextArgs,
        )
    )

    tools.append(
        StructuredTool(
            name="workspace__search",
            description=(
                "[Workspace] Search across all sources and notes in the workspace "
                "using semantic search. Returns matching documents with relevance "
                "scores. Use this to find existing content before creating duplicates."
            ),
            func=_search_workspace,
            args_schema=SearchWorkspaceArgs,
        )
    )

    return tools
