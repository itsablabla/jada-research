# Testing MCP Integration

End-to-end testing procedure for the MCP (Model Context Protocol) tool integration feature in Open Notebook.

## App URL

- Production: https://research.garzaos.online
- Embedded in Nextcloud: https://next.garzaos.online/apps/external/4
- API base: https://research.garzaos.online/api

## Devin Secrets Needed

- SSH key for deployment server (83.228.213.100, user: `ubuntu`) — needed for redeployment
- App password: `open-notebook-change-me` (default dev password, used in `Authorization: Bearer` header)

## Deployment

- Server: 83.228.213.100 (SSH as `ubuntu`)
- Path: `/opt/jada-research/repo` (git repo) and `/opt/jada-research/docker-compose.yml`
- Container: `jada-research` (app), `jada-research-db` (SurrealDB v2)
- Rebuild: `cd /opt/jada-research/repo && git pull && cd /opt/jada-research && docker compose build jada-research && docker compose up -d jada-research`
- Verify: `curl -s -H 'Authorization: Bearer open-notebook-change-me' https://research.garzaos.online/api/mcp/servers | python3 -m json.tool`

## Test Plan (5 Tests)

### Test 1: MCP Settings Page
1. Navigate to `/settings/mcp`
2. Assert: Page title "MCP Tools", 2 server cards (Composio + Garza MCP), both "Active"
3. Assert: `GET /api/mcp/servers` returns masked headers (`****last4` format, not plaintext)

### Test 2: Edit Server Dialog Safety
1. Click Edit (pencil) on any server card
2. Assert: Headers textarea shows empty `{ }`, NOT masked `****` values
3. Assert: Placeholder says "Leave empty to keep existing headers"
4. Cancel without saving
5. Verify credentials not destroyed: re-check API returns same masked values

### Test 3: Connection Test
1. Click test (plug) icon on a server card
2. Assert: Loading spinner appears, page stays responsive
3. Assert: Result shows green "Connected — N tools available" or red error
4. Assert: No 500 error or page freeze

### Test 4: Chat with MCP Tools
1. Navigate to a notebook with chat history (e.g., "Garza Research Test")
2. Send a message triggering tool use: "Use one of your tools to tell me the current date and time"
3. Assert: AI responds with readable text (not raw errors)
4. Assert: No `"content Input should be a valid string"` validation error
5. Assert: No StructuredTool serialization crash
6. Assert: No `INVALID_ARGUMENT` Gemini schema error

### Test 5: Sidebar i18n
1. Check sidebar shows "MCP Tools" under "Manage" (English)
2. Switch language to Português via Language button
3. Assert: Label changes to "Ferramentas MCP" under "Gerenciar"
4. Switch back to English

## Known Gotchas

- **Composio tool call formatting**: Composio's `COMPOSIO_MULTI_EXECUTE_TOOL` endpoint may return validation errors (`"Expected object, received string"`) for some tool call argument formats. This is a Composio API-side issue, not a bug in the app. The AI should gracefully handle this.
- **Connection test latency**: The MCP connection test can take 10-20 seconds for remote servers. Wait patiently.
- **Gemini content format**: Gemini returns content as `[{'text': '...', 'type': 'direct'}]` list format. The `_normalize_content()` helper converts this to a plain string. If you see raw list-of-dicts in chat responses, the normalization might not be deployed.
- **Edit dialog credential destruction**: If the edit dialog pre-fills headers with `****SzJZ` masked values and user saves, real credentials get overwritten. The fix tracks `headersModified` state and only sends headers if explicitly changed.
- **Tool round limiting**: `should_continue()` in `chat.py` counts tool rounds only since the last HumanMessage. If tools seem permanently disabled in a thread, check if the counting logic regressed to full-history counting.
- **StructuredTool serialization**: LangGraph's SqliteSaver cannot serialize StructuredTool objects with closures. `mcp_tools` must NOT be stored in ThreadState. If chat crashes with `TypeError: cannot pickle`, check if someone re-added it.
- **Schema validation for Gemini**: Gemini rejects tool schemas with `Optional[T]`/`anyOf`, empty `properties`, or missing `items.type` on arrays. The `_build_schema()` function in `langchain_bridge.py` sanitizes these.
- **API auth for curl tests**: Always include `Authorization: Bearer open-notebook-change-me` header.
- **Deployment verification**: After deploying, always verify with a curl to `/api/mcp/servers` to confirm the new code is live. Check that headers are masked (if they show plaintext, the masking fix isn't deployed).
