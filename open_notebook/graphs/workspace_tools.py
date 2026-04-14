"""
Native workspace tools for the chat AI.

These LangChain StructuredTools let the chat agent interact with the
Open Notebook workspace directly — creating notes, adding URL sources,
searching existing content, and searching emails via IMAP.  They are
merged with MCP tools in the chat graph so the AI can seamlessly use
both external services and its own workspace.
"""

import asyncio
import concurrent.futures
import email
import email.header
import email.utils
import imaplib
import os
import re
import ssl
from datetime import datetime, timedelta
from functools import partial
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
    # Use str with default="" instead of Optional[str] to avoid anyOf in JSON
    # schema which Gemini's function-calling API rejects with INVALID_ARGUMENT.
    notebook_id: str = Field(
        default="",
        description="The notebook ID to save the note to (e.g. 'notebook:abc123'). "
        "Leave empty to use the current notebook."
    )


class AddSourceFromURLArgs(BaseModel):
    url: str = Field(description="The URL to ingest as a new source")
    notebook_id: str = Field(
        default="",
        description="The notebook ID to add the source to (e.g. 'notebook:abc123'). "
        "Leave empty to use the current notebook."
    )


class AddSourceFromTextArgs(BaseModel):
    title: str = Field(description="Title for the text source")
    text: str = Field(description="The full text content to save as a source")
    notebook_id: str = Field(
        default="",
        description="The notebook ID to add the source to (e.g. 'notebook:abc123'). "
        "Leave empty to use the current notebook."
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
# IMAP Email helpers
# ---------------------------------------------------------------------------

def _get_imap_config():
    """Read IMAP connection settings from environment."""
    return {
        "host": os.environ.get("IMAP_HOST", ""),
        "port": int(os.environ.get("IMAP_PORT", "11143")),
        "user": os.environ.get("IMAP_USER", ""),
        "password": os.environ.get("IMAP_PASSWORD", ""),
        "use_ssl": os.environ.get("IMAP_USE_SSL", "false").lower() == "true",
        "use_starttls": os.environ.get("IMAP_USE_STARTTLS", "true").lower() == "true",
    }


def _decode_header_value(raw: str) -> str:
    """Decode an RFC-2047 encoded header into a plain string."""
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    decoded = []
    for data, charset in parts:
        if isinstance(data, bytes):
            decoded.append(data.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(data)
    return " ".join(decoded)


def _extract_text_from_email(msg: email.message.Message) -> str:
    """Extract plain-text body from an email message."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
            elif ct == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    html = payload.decode(charset, errors="replace")
                    # Strip HTML tags for a rough plain-text conversion
                    text = re.sub(r"<[^>]+>", " ", html)
                    text = re.sub(r"\s+", " ", text).strip()
                    return text
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


def _imap_connect(timeout: int = 30):
    """Open an IMAP connection using env config with a socket timeout.

    Supports three TLS modes (checked in order):
    1. ``IMAP_USE_SSL=true``  → connect via ``IMAP4_SSL`` (implicit TLS)
    2. ``IMAP_USE_STARTTLS=true`` (default) → plain connect then STARTTLS upgrade
    3. Both false → plain-text (not recommended)

    ProtonMail Bridge requires STARTTLS on its non-SSL IMAP port.
    """
    cfg = _get_imap_config()
    if not cfg["host"] or not cfg["user"]:
        raise ValueError(
            "IMAP is not configured. Set IMAP_HOST, IMAP_USER, and IMAP_PASSWORD."
        )
    if cfg["use_ssl"]:
        conn = imaplib.IMAP4_SSL(cfg["host"], cfg["port"], timeout=timeout)
    else:
        conn = imaplib.IMAP4(cfg["host"], cfg["port"], timeout=timeout)
        if cfg["use_starttls"]:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            conn.starttls(ssl_context=ctx)
            logger.debug("IMAP STARTTLS upgrade successful")
    conn.login(cfg["user"], cfg["password"])
    return conn


def _imap_search(query: str, folder: str = "INBOX", max_results: int = 20) -> list:
    """Fetch recent emails via IMAP and filter client-side by keyword.

    ProtonMail Bridge's server-side SEARCH is extremely slow on large
    mailboxes (decrypts every message). Instead we fetch the most recent
    ``scan_count`` messages by sequence number and filter locally.
    """
    # Scan more messages than requested so filtering still yields enough.
    # Keep scan_count modest — each message takes ~0.4s on ProtonMail Bridge.
    scan_count = min(max(max_results * 2, 30), 50)

    conn = _imap_connect(timeout=120)
    try:
        status, data = conn.select(folder, readonly=True)
        total = int(data[0])
        if total == 0:
            return []

        # Sequence range: fetch the last `scan_count` messages
        start = max(1, total - scan_count + 1)
        seq_range = f"{start}:{total}"

        # Fetch headers + body in one call (faster than individual fetches)
        _status, fetch_data = conn.fetch(seq_range, "(RFC822)")

        # Parse fetched messages
        raw_messages = []
        for item in fetch_data:
            if isinstance(item, tuple) and len(item) == 2:
                raw_messages.append(item[1])

        query_lower = query.strip().lower()
        results = []

        # Process newest first
        for raw_email in reversed(raw_messages):
            if len(results) >= max_results:
                break
            try:
                msg = email.message_from_bytes(raw_email)
                subject = _decode_header_value(msg.get("Subject", ""))
                from_raw = msg.get("From", "")
                from_name, from_addr = email.utils.parseaddr(from_raw)
                from_name = _decode_header_value(from_name) or from_addr
                date_str = msg.get("Date", "")
                body = _extract_text_from_email(msg)

                # Client-side keyword filter
                if query_lower:
                    searchable = f"{subject} {from_name} {from_addr} {body}".lower()
                    if query_lower not in searchable:
                        continue

                body_preview = body[:500] + "..." if len(body) > 500 else body

                results.append({
                    "uid": "0",
                    "subject": subject,
                    "from_name": from_name,
                    "from_email": from_addr,
                    "date": date_str,
                    "body_preview": body_preview,
                    "body_full": body,
                })
            except Exception as parse_err:
                logger.debug(f"Skipping unparseable email: {parse_err}")
                continue

        return results
    finally:
        try:
            conn.logout()
        except Exception:
            pass


class SearchEmailsArgs(BaseModel):
    query: str = Field(description="Search term to find in email subjects and bodies")
    max_results: int = Field(
        default=10, description="Maximum number of emails to return (newest first, max 20)"
    )
    folder: str = Field(
        default="INBOX", description="IMAP folder to search (default: INBOX)"
    )


class SearchAndSaveEmailsArgs(BaseModel):
    query: str = Field(description="Search term to find in email subjects and bodies")
    max_results: int = Field(
        default=10, description="Maximum number of emails to search and save (newest first, max 20)"
    )
    folder: str = Field(
        default="INBOX", description="IMAP folder to search (default: INBOX)"
    )
    save_as: str = Field(
        default="source",
        description="How to save: 'source' (searchable source) or 'note' (note)"
    )
    notebook_id: str = Field(
        default="",
        description="The notebook ID to save emails to. Leave empty for current notebook."
    )


def _search_emails(query: str, max_results: int = 20, folder: str = "INBOX") -> str:
    """Search emails via IMAP and return summaries."""
    logger.info(f"Workspace tool search_emails called: query='{query}', max_results={max_results}, folder='{folder}'")
    try:
        results = _imap_search(query, folder, max_results)
        if not results:
            return f"No emails found matching '{query}' in {folder}."

        lines = [f"Found {len(results)} email(s) matching '{query}':\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"**{i}. {r['subject']}**\n"
                f"   From: {r['from_name']} <{r['from_email']}>\n"
                f"   Date: {r['date']}\n"
                f"   Preview: {r['body_preview'][:200]}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Workspace tool search_emails failed: {e}", exc_info=True)
        return f"Error searching emails: {e}"


def _search_and_save_emails(
    query: str,
    max_results: int = 20,
    folder: str = "INBOX",
    save_as: str = "source",
    notebook_id: str = "",
) -> str:
    """Search emails via IMAP and save each one to the notebook."""
    logger.info(
        f"Workspace tool search_and_save_emails called: query='{query}', "
        f"max_results={max_results}, save_as='{save_as}', notebook_id='{notebook_id}'"
    )
    try:
        results = _imap_search(query, folder, max_results)
        if not results:
            return f"No emails found matching '{query}' in {folder}."

        saved = []
        errors = []

        for r in results:
            title = f"{r['subject']} — from {r['from_name']} ({r['date'][:16]})"
            content = (
                f"**From:** {r['from_name']} <{r['from_email']}>\n"
                f"**Date:** {r['date']}\n"
                f"**Subject:** {r['subject']}\n\n"
                f"---\n\n{r['body_full']}"
            )

            try:
                if save_as == "note":
                    result_msg = _create_note(
                        title=title, content=content, notebook_id=notebook_id
                    )
                else:
                    result_msg = _add_source_from_text(
                        title=title, text=content, notebook_id=notebook_id
                    )
                saved.append(f"- {title}")
                logger.info(f"Saved email as {save_as}: {title}")
            except Exception as e:
                errors.append(f"- {r['subject']}: {e}")
                logger.error(f"Failed to save email '{r['subject']}': {e}")

        summary = [f"Searched for '{query}' — found {len(results)} emails.\n"]
        if saved:
            summary.append(f"**Saved {len(saved)} emails as {save_as}s:**")
            summary.extend(saved)
        if errors:
            summary.append(f"\n**{len(errors)} error(s):**")
            summary.extend(errors)
        return "\n".join(summary)
    except Exception as e:
        logger.error(f"Workspace tool search_and_save_emails failed: {e}", exc_info=True)
        return f"Error searching/saving emails: {e}"


# ---------------------------------------------------------------------------
# Public API — returns the list of workspace StructuredTools
# ---------------------------------------------------------------------------

def _with_default_notebook(func, default_notebook_id: str):
    """Wrap a tool function so notebook_id defaults to default_notebook_id when not provided."""
    def wrapper(*args, **kwargs):
        # Treat both None and empty string as "not provided"
        if not kwargs.get("notebook_id"):
            kwargs["notebook_id"] = default_notebook_id
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def get_workspace_tools(notebook_id: Optional[str] = None) -> List[StructuredTool]:
    """
    Build the list of native workspace tools.

    If *notebook_id* is provided, the create/add tools will default to that
    notebook — the AI doesn't need to specify it explicitly.
    """
    tools: List[StructuredTool] = []

    # Bind notebook_id as default if available
    create_note_fn = _with_default_notebook(_create_note, notebook_id) if notebook_id else _create_note
    add_url_fn = _with_default_notebook(_add_source_from_url, notebook_id) if notebook_id else _add_source_from_url
    add_text_fn = _with_default_notebook(_add_source_from_text, notebook_id) if notebook_id else _add_source_from_text

    nb_hint = f" (defaults to current notebook {notebook_id})" if notebook_id else ""

    tools.append(
        StructuredTool(
            name="workspace__create_note",
            description=(
                "[Workspace] Create a new note in the current notebook. "
                "Use this to save summaries, observations, extracted data, "
                "or any other text the user wants to keep as a note."
                f"{nb_hint}"
            ),
            func=create_note_fn,
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
                f"{nb_hint}"
            ),
            func=add_url_fn,
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
                f"{nb_hint}"
            ),
            func=add_text_fn,
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

    # Email tools (only available when IMAP is configured)
    imap_cfg = _get_imap_config()
    if imap_cfg["host"] and imap_cfg["user"]:
        tools.append(
            StructuredTool(
                name="workspace__search_emails",
                description=(
                    "[Email] Search your email inbox via IMAP. Returns email subjects, "
                    "senders, dates, and body previews. Use this to find emails "
                    "matching a search term."
                ),
                func=_search_emails,
                args_schema=SearchEmailsArgs,
            )
        )

        save_emails_fn = (
            _with_default_notebook(_search_and_save_emails, notebook_id)
            if notebook_id
            else _search_and_save_emails
        )
        tools.append(
            StructuredTool(
                name="workspace__search_and_save_emails",
                description=(
                    "[Email] Search your email inbox and save ALL matching emails "
                    "to the current notebook in one step. Each email becomes a "
                    "separate source (or note). This is the preferred tool when "
                    "the user asks to 'find and save emails' or 'import emails'. "
                    "Saves happen in bulk — no need to call create_note separately "
                    "for each email."
                    f"{nb_hint}"
                ),
                func=save_emails_fn,
                args_schema=SearchAndSaveEmailsArgs,
            )
        )
        logger.info("Email tools enabled (IMAP configured)")
    else:
        logger.debug("Email tools disabled (IMAP not configured)")

    return tools
