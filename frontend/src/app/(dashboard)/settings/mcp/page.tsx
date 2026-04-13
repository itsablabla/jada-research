'use client'

import { useState } from 'react'
import { AppShell } from '@/components/layout/AppShell'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Label } from '@/components/ui/label'
import {
  RefreshCw,
  Plus,
  Edit,
  Trash2,
  Loader2,
  Plug,
  Unplug,
  Wrench,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import {
  useMCPServers,
  useAddMCPServer,
  useUpdateMCPServer,
  useDeleteMCPServer,
  useTestMCPConnection,
} from '@/lib/hooks/use-mcp'
import { MCPServer, MCPConnectionTestResult } from '@/lib/api/mcp'

// =============================================================================
// Add/Edit Server Dialog
// =============================================================================

function ServerFormDialog({
  open,
  onOpenChange,
  server,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  server?: MCPServer | null
}) {
  const addServer = useAddMCPServer()
  const updateServer = useUpdateMCPServer()
  const isEditing = !!server
  const isSubmitting = addServer.isPending || updateServer.isPending

  const [name, setName] = useState(server?.name || '')
  const [url, setUrl] = useState(server?.url || '')
  const [headersText, setHeadersText] = useState(
    server?.headers ? JSON.stringify(server.headers, null, 2) : '{\n  \n}'
  )
  const [description, setDescription] = useState(server?.description || '')
  const [enabled, setEnabled] = useState(server?.enabled ?? true)
  const [headersError, setHeadersError] = useState<string | null>(null)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    let headers: Record<string, string> = {}
    try {
      headers = JSON.parse(headersText)
      setHeadersError(null)
    } catch {
      setHeadersError('Invalid JSON')
      return
    }

    const onSuccess = () => onOpenChange(false)

    if (isEditing && server) {
      updateServer.mutate(
        {
          serverId: server.id,
          data: { name, url, headers, description: description || undefined, enabled },
        },
        { onSuccess }
      )
    } else {
      addServer.mutate(
        { name, url, headers, description: description || undefined, enabled },
        { onSuccess }
      )
    }
  }

  const isValid = name.trim() !== '' && url.trim() !== ''

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{isEditing ? 'Edit MCP Server' : 'Add MCP Server'}</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="server-name">Name</Label>
            <input
              id="server-name"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Composio, Garza MCP"
              disabled={isSubmitting}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="server-url">URL</Label>
            <input
              id="server-url"
              type="url"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/mcp"
              disabled={isSubmitting}
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="server-headers">Headers (JSON)</Label>
            <textarea
              id="server-headers"
              className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
              value={headersText}
              onChange={(e) => {
                setHeadersText(e.target.value)
                setHeadersError(null)
              }}
              placeholder='{"Authorization": "Bearer ..."}'
              disabled={isSubmitting}
            />
            {headersError && (
              <p className="text-xs text-destructive">{headersError}</p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="server-description">Description (optional)</Label>
            <input
              id="server-description"
              className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What tools does this server provide?"
              disabled={isSubmitting}
            />
          </div>

          <div className="flex items-center gap-2">
            <input
              id="server-enabled"
              type="checkbox"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
              className="h-4 w-4 rounded border-input"
              disabled={isSubmitting}
            />
            <Label htmlFor="server-enabled">Enabled</Label>
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)} disabled={isSubmitting}>
              Cancel
            </Button>
            <Button type="submit" disabled={!isValid || isSubmitting}>
              {isSubmitting && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
              {isEditing ? 'Save' : 'Add Server'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

// =============================================================================
// Server Card
// =============================================================================

function ServerCard({
  server,
  onEdit,
  onDelete,
}: {
  server: MCPServer
  onEdit: () => void
  onDelete: () => void
}) {
  const testConnection = useTestMCPConnection()
  const [testResult, setTestResult] = useState<MCPConnectionTestResult | null>(null)
  const [showTools, setShowTools] = useState(false)

  const handleTest = () => {
    setTestResult(null)
    testConnection.mutate(server.id, {
      onSuccess: (result) => setTestResult(result),
    })
  }

  return (
    <Card className={!server.enabled ? 'opacity-60' : ''}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-lg">{server.name}</CardTitle>
            {server.enabled ? (
              <Badge variant="default" className="bg-green-600">Active</Badge>
            ) : (
              <Badge variant="secondary">Disabled</Badge>
            )}
          </div>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={handleTest} disabled={testConnection.isPending}>
              {testConnection.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Plug className="h-4 w-4" />
              )}
            </Button>
            <Button variant="ghost" size="sm" onClick={onEdit}>
              <Edit className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={onDelete} className="text-destructive hover:text-destructive">
              <Trash2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <CardDescription className="text-xs font-mono truncate">{server.url}</CardDescription>
        {server.description && (
          <p className="text-sm text-muted-foreground">{server.description}</p>
        )}
      </CardHeader>
      <CardContent>
        {testResult && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              {testResult.status === 'connected' ? (
                <>
                  <Plug className="h-4 w-4 text-green-600" />
                  <span className="text-sm text-green-600 font-medium">
                    Connected — {testResult.tool_count} tools available
                  </span>
                </>
              ) : (
                <>
                  <Unplug className="h-4 w-4 text-destructive" />
                  <span className="text-sm text-destructive font-medium">
                    Failed: {testResult.error}
                  </span>
                </>
              )}
            </div>

            {testResult.status === 'connected' && testResult.tools.length > 0 && (
              <div>
                <button
                  onClick={() => setShowTools(!showTools)}
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
                >
                  {showTools ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                  {showTools ? 'Hide tools' : `Show ${testResult.tools.length} tools`}
                </button>
                {showTools && (
                  <div className="mt-2 max-h-48 overflow-y-auto rounded border p-2">
                    <div className="flex flex-wrap gap-1">
                      {testResult.tools.map((tool) => (
                        <Badge key={tool} variant="outline" className="text-xs font-mono">
                          {tool}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// =============================================================================
// Main Page
// =============================================================================

export default function MCPSettingsPage() {
  const { data: servers, isLoading, refetch } = useMCPServers()
  const deleteServer = useDeleteMCPServer()

  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [editServer, setEditServer] = useState<MCPServer | null>(null)
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null)

  const handleDelete = (serverId: string) => {
    deleteServer.mutate(serverId, {
      onSuccess: () => setDeleteConfirmId(null),
    })
  }

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6">
          <div className="max-w-4xl">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-4">
                <Wrench className="h-6 w-6" />
                <h1 className="text-2xl font-bold">MCP Tools</h1>
                <Button variant="outline" size="sm" onClick={() => refetch()}>
                  <RefreshCw className="h-4 w-4" />
                </Button>
              </div>
              <Button onClick={() => setAddDialogOpen(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Add Server
              </Button>
            </div>

            <p className="text-sm text-muted-foreground mb-6">
              Connect external MCP (Model Context Protocol) servers to give the AI access to
              external tools during chat conversations. Tools from connected servers are
              automatically available when chatting in notebooks.
            </p>

            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : !servers || servers.length === 0 ? (
              <Card className="border-dashed">
                <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                  <Wrench className="h-12 w-12 text-muted-foreground mb-4" />
                  <h3 className="text-lg font-medium mb-2">No MCP servers configured</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Add an MCP server to give the AI access to external tools like
                    email, file management, and more.
                  </p>
                  <Button onClick={() => setAddDialogOpen(true)}>
                    <Plus className="h-4 w-4 mr-2" />
                    Add Your First Server
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-4">
                {servers.map((server) => (
                  <ServerCard
                    key={server.id}
                    server={server}
                    onEdit={() => setEditServer(server)}
                    onDelete={() => setDeleteConfirmId(server.id)}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Add Dialog */}
      <ServerFormDialog
        open={addDialogOpen}
        onOpenChange={(open) => {
          setAddDialogOpen(open)
        }}
      />

      {/* Edit Dialog */}
      {editServer && (
        <ServerFormDialog
          open={!!editServer}
          onOpenChange={(open) => {
            if (!open) setEditServer(null)
          }}
          server={editServer}
        />
      )}

      {/* Delete Confirmation */}
      <Dialog open={!!deleteConfirmId} onOpenChange={(open) => { if (!open) setDeleteConfirmId(null) }}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader>
            <DialogTitle>Delete MCP Server?</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            This will remove the server and its tools will no longer be available during chat.
          </p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeleteConfirmId(null)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={() => deleteConfirmId && handleDelete(deleteConfirmId)}
              disabled={deleteServer.isPending}
            >
              {deleteServer.isPending && <Loader2 className="h-4 w-4 animate-spin mr-2" />}
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </AppShell>
  )
}
