import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listMCPServers,
  addMCPServer,
  updateMCPServer,
  deleteMCPServer,
  testMCPConnection,
  listMCPServerTools,
  AddMCPServerRequest,
  UpdateMCPServerRequest,
} from '@/lib/api/mcp'

export function useMCPServers() {
  return useQuery({
    queryKey: ['mcp-servers'],
    queryFn: listMCPServers,
  })
}

export function useAddMCPServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (request: AddMCPServerRequest) => addMCPServer(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mcp-servers'] })
    },
  })
}

export function useUpdateMCPServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ serverId, data }: { serverId: string; data: UpdateMCPServerRequest }) =>
      updateMCPServer(serverId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mcp-servers'] })
    },
  })
}

export function useDeleteMCPServer() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (serverId: string) => deleteMCPServer(serverId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mcp-servers'] })
    },
  })
}

export function useTestMCPConnection() {
  return useMutation({
    mutationFn: (serverId: string) => testMCPConnection(serverId),
  })
}

export function useMCPServerTools(serverId: string | null) {
  return useQuery({
    queryKey: ['mcp-server-tools', serverId],
    queryFn: () => listMCPServerTools(serverId!),
    enabled: !!serverId,
  })
}
