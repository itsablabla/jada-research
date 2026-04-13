import { apiClient } from './client'

// Types
export interface MCPServer {
  id: string
  name: string
  url: string
  headers: Record<string, string>
  enabled: boolean
  description?: string | null
}

export interface AddMCPServerRequest {
  name: string
  url: string
  headers?: Record<string, string>
  enabled?: boolean
  description?: string | null
}

export interface UpdateMCPServerRequest {
  name?: string
  url?: string
  headers?: Record<string, string>
  enabled?: boolean
  description?: string | null
}

export interface MCPToolInfo {
  name: string
  description?: string | null
  input_schema?: Record<string, unknown> | null
}

export interface MCPConnectionTestResult {
  status: string
  tool_count: number
  tools: string[]
  error?: string | null
}

// API functions
export async function listMCPServers(): Promise<MCPServer[]> {
  const { data } = await apiClient.get('/mcp/servers')
  return data
}

export async function addMCPServer(request: AddMCPServerRequest): Promise<MCPServer> {
  const { data } = await apiClient.post('/mcp/servers', request)
  return data
}

export async function updateMCPServer(serverId: string, request: UpdateMCPServerRequest): Promise<MCPServer> {
  const { data } = await apiClient.put(`/mcp/servers/${serverId}`, request)
  return data
}

export async function deleteMCPServer(serverId: string): Promise<void> {
  await apiClient.delete(`/mcp/servers/${serverId}`)
}

export async function testMCPConnection(serverId: string): Promise<MCPConnectionTestResult> {
  const { data } = await apiClient.post(`/mcp/servers/${serverId}/test`)
  return data
}

export async function listMCPServerTools(serverId: string): Promise<MCPToolInfo[]> {
  const { data } = await apiClient.get(`/mcp/servers/${serverId}/tools`)
  return data
}
