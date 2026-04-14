# Testing Open Notebook App

Procedures for testing the Open Notebook research assistant at https://research.garzaos.online

## Devin Secrets Needed

- SSH key for deployment server (83.228.213.100) — provided as attachment, not a stored secret
- `OPENAI_API_KEY` — needed if registering OpenAI models (text-embedding-3-small)
- App password for API auth (default: `open-notebook-change-me`, check `OPEN_NOTEBOOK_PASSWORD`)

## App Access

- **URL**: https://research.garzaos.online
- **API**: https://research.garzaos.online/api (port 5055 internally)
- **Auth**: Password-based via `Authorization: Bearer <password>` header
- **Default password**: `open-notebook-change-me` (set via `OPEN_NOTEBOOK_PASSWORD` env var)
- The app may also be accessed via Nextcloud iframe at https://next.garzaos.online/apps/external/4

## Deployment

- **Server**: 83.228.213.100 (SSH with key, user `root`)
- **Container**: `open_notebook` (Docker Compose service)
- **Deploy command**: `cd /root/jada-research && git pull && docker compose build open_notebook && docker compose up -d open_notebook`
- **DB container**: `jada-research-db` (SurrealDB) — do NOT restart unless necessary
- **Logs**: `docker logs open_notebook --tail 100 -f`

## Model Configuration via API

Default models are configured via the API, not code changes.

```bash
# Get current defaults
curl -s https://research.garzaos.online/api/models/defaults \
  -H "Authorization: Bearer open-notebook-change-me" | jq

# List all registered models
curl -s https://research.garzaos.online/api/models \
  -H "Authorization: Bearer open-notebook-change-me" | jq

# Update defaults (PATCH with model IDs)
curl -X PATCH https://research.garzaos.online/api/models/defaults \
  -H "Authorization: Bearer open-notebook-change-me" \
  -H "Content-Type: application/json" \
  -d '{"default_chat_model": "model:<id>", "default_transformation_model": "model:<id>"}'
```

Model IDs are in format `model:<random_id>`. Get IDs from the list models endpoint.

## Testing Chat (Notebook)

1. Navigate to https://research.garzaos.online/notebooks
2. Click on a notebook (e.g., "Garza Research Test")
3. Chat panel is on the right side with textarea at bottom
4. **Enter sends message** (not Ctrl+Enter) — Shift+Enter adds newline
5. Model selector button shows current model name next to gear icon
6. Click model selector to open dialog with dropdown of all language models
7. "Default" option maps to whatever `default_chat_model` is configured

### Key assertions for chat testing:
- Placeholder text should say "Press Enter to send"
- Message appears as blue bubble (right-aligned) after Enter
- Textarea clears and disables during processing
- AI response appears as grey bubble (left-aligned)
- No "content Input should be a valid string" errors (Gemini content normalization)
- No "INVALID_ARGUMENT" errors (Gemini schema validation)

## Testing Search Page

1. Navigate to https://research.garzaos.online/search
2. "Ask (beta)" tab is selected by default
3. Textarea with "Enter your question..." placeholder
4. **Hint text below textarea says "Press Enter to submit"**
5. Enter submits the query (shows "Processing..." spinner)
6. Model badges show Strategy/Answer/Final model names

## Testing MCP Tools

1. Navigate to Settings > MCP Tools (https://research.garzaos.online/settings/mcp)
2. Server cards show name, URL, status (Active/Disabled)
3. **Headers are masked** in API responses (e.g., `****SzJZ`)
4. Edit dialog should NOT pre-fill masked headers — textarea should be empty with placeholder "Leave empty to keep existing headers"
5. Test connection button verifies server connectivity
6. "Show Tools" expands tool list

### MCP Chat Testing:
- Use a notebook with MCP tools enabled
- Send message like "Use your Garza MCP tools to list the notes in Nextcloud"
- AI should call MCP tools and return real data
- Watch for serialization errors (StructuredTool objects in state)

## Common Issues

- **Gemini content validation**: Gemini returns content as `[{'text': '...'}]` instead of string. The `_normalize_content()` helper in `api/routers/chat.py` and `source_chat.py` fixes this.
- **Gemini schema errors**: Empty `properties` in tool schemas causes `INVALID_ARGUMENT`. The `langchain_bridge.py` sanitizer adds a placeholder property.
- **Tool round limits**: `MAX_TOOL_ROUNDS` (default 10) counts only since last HumanMessage, not entire history.
- **SSH key not persisted**: The SSH key for 83.228.213.100 is provided as an attachment and may not carry over between sessions. Request it from the user if needed.
- **i18n**: All UI string changes must be reflected in 9 locale files (en-US, pt-BR, zh-CN, zh-TW, ja-JP, ru-RU, bn-IN, fr-FR, it-IT).
