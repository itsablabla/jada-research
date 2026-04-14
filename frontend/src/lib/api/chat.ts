import apiClient from './client'
import { getApiUrl } from '@/lib/config'
import {
  NotebookChatSession,
  NotebookChatSessionWithMessages,
  CreateNotebookChatSessionRequest,
  UpdateNotebookChatSessionRequest,
  SendNotebookChatMessageRequest,
  NotebookChatMessage,
  BuildContextRequest,
  BuildContextResponse,
} from '@/lib/types/api'

/**
 * Get the auth token from localStorage (same source as axios interceptor).
 */
function getAuthToken(): string | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = localStorage.getItem('auth-storage')
    if (!raw) return null
    const { state } = JSON.parse(raw)
    return state?.token ?? null
  } catch {
    return null
  }
}

export const chatApi = {
  // Session management
  listSessions: async (notebookId: string) => {
    const response = await apiClient.get<NotebookChatSession[]>(
      `/chat/sessions`,
      { params: { notebook_id: notebookId } }
    )
    return response.data
  },

  createSession: async (data: CreateNotebookChatSessionRequest) => {
    const response = await apiClient.post<NotebookChatSession>(
      `/chat/sessions`,
      data
    )
    return response.data
  },

  getSession: async (sessionId: string) => {
    const response = await apiClient.get<NotebookChatSessionWithMessages>(
      `/chat/sessions/${sessionId}`
    )
    return response.data
  },

  updateSession: async (sessionId: string, data: UpdateNotebookChatSessionRequest) => {
    const response = await apiClient.put<NotebookChatSession>(
      `/chat/sessions/${sessionId}`,
      data
    )
    return response.data
  },

  deleteSession: async (sessionId: string) => {
    await apiClient.delete(`/chat/sessions/${sessionId}`)
  },

  /**
   * Send a chat message using SSE streaming.
   *
   * The backend sends heartbeat events every ~10s to keep the nginx
   * connection alive while the LangGraph chat graph processes multi-round
   * tool calling. The final result arrives as a "complete" event.
   *
   * Falls back to the non-streaming endpoint if SSE fails to connect.
   */
  sendMessage: async (data: SendNotebookChatMessageRequest): Promise<{
    session_id: string
    messages: NotebookChatMessage[]
  }> => {
    const baseUrl = await getApiUrl()
    const token = getAuthToken()

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }
    if (token) {
      headers['Authorization'] = `Bearer ${token}`
    }

    // Use the SSE streaming endpoint
    const response = await fetch(`${baseUrl}/api/chat/execute/stream`, {
      method: 'POST',
      headers,
      body: JSON.stringify(data),
    })

    if (!response.ok) {
      // Try to parse error detail from response
      let detail = `HTTP ${response.status}`
      try {
        const errBody = await response.json()
        detail = errBody.detail || detail
      } catch {
        // ignore parse errors
      }
      throw { response: { data: { detail } }, message: detail }
    }

    if (!response.body) {
      throw { message: 'No response body from streaming endpoint' }
    }

    // Parse SSE stream
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let result: { session_id: string; messages: NotebookChatMessage[] } | null = null

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // Process complete SSE events (separated by \n\n)
      const parts = buffer.split('\n\n')
      buffer = parts.pop() || '' // Keep incomplete last part

      for (const part of parts) {
        const trimmed = part.trim()
        if (!trimmed.startsWith('data: ')) continue

        try {
          const eventData = JSON.parse(trimmed.slice(6))

          if (eventData.type === 'complete') {
            result = {
              session_id: eventData.session_id,
              messages: eventData.messages,
            }
          } else if (eventData.type === 'error') {
            throw {
              response: { data: { detail: eventData.detail } },
              message: eventData.detail,
            }
          }
          // heartbeat events are silently ignored
        } catch (parseErr) {
          // If it's our own thrown error, re-throw
          if (parseErr && typeof parseErr === 'object' && 'response' in parseErr) {
            throw parseErr
          }
          // Otherwise ignore malformed SSE lines
          console.warn('Failed to parse SSE event:', trimmed)
        }
      }
    }

    if (!result) {
      throw { message: 'Stream ended without a complete event' }
    }

    return result
  },

  buildContext: async (data: BuildContextRequest) => {
    const response = await apiClient.post<BuildContextResponse>(
      `/chat/context`,
      data
    )
    return response.data
  },
}

export default chatApi
