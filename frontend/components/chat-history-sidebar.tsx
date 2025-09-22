"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { MessageSquare, Plus, Search, Trash2, Clock, MoreVertical } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"

export type ChatSession = {
  id: string
  title: string
  preview: string
  timestamp: Date
  messageCount: number
}

interface ChatHistorySidebarProps {
  sessions: ChatSession[]
  currentSessionId: string | null
  onSelectSession: (sessionId: string) => void
  onNewChat: () => void
  onDeleteSession: (sessionId: string) => void
}

export function ChatHistorySidebar({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
}: ChatHistorySidebarProps) {
  const router = useRouter()
  const [searchQuery, setSearchQuery] = useState("")

  const filteredSessions = sessions.filter(
    (session) =>
      session.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      session.preview.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const formatDate = (date: Date) => {
    const now = new Date()
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate())
    const yesterday = new Date(today)
    yesterday.setDate(yesterday.getDate() - 1)

    const sessionDate = new Date(date.getFullYear(), date.getMonth(), date.getDate())

    if (sessionDate.getTime() === today.getTime()) {
      return "Today"
    } else if (sessionDate.getTime() === yesterday.getTime()) {
      return "Yesterday"
    } else {
      return date.toLocaleDateString(undefined, { month: "short", day: "numeric" })
    }
  }

  return (
    <div className="flex h-full w-[280px] flex-col border-r">
      <div className="flex items-center justify-between p-4 border-b">
        <h2 className="text-lg font-semibold">Chat History</h2>
        <Button variant="outline" size="icon" onClick={onNewChat} title="New Chat">
          <Plus className="h-4 w-4" />
        </Button>
      </div>

      <div className="p-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search chats..."
            className="pl-8"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="space-y-1 p-2">
          {filteredSessions.length > 0 ? (
            filteredSessions.map((session) => (
              <div
                key={session.id}
                className={`group flex cursor-pointer flex-col rounded-md p-3 transition-colors ${
                  currentSessionId === session.id ? "bg-accent text-accent-foreground" : "hover:bg-muted"
                }`}
                onClick={() => onSelectSession(session.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4" />
                    <span className="font-medium truncate">{session.title}</span>
                  </div>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 opacity-0 group-hover:opacity-100"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <MoreVertical className="h-3.5 w-3.5" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        className="text-destructive focus:text-destructive"
                        onClick={(e) => {
                          e.stopPropagation()
                          onDeleteSession(session.id)
                        }}
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                <div className="mt-1 flex items-center justify-between">
                  <span className="text-xs text-muted-foreground truncate max-w-[160px]">{session.preview}</span>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    <span>{formatDate(session.timestamp)}</span>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <MessageSquare className="h-8 w-8 text-muted-foreground mb-2" />
              <h3 className="text-sm font-medium">No chats found</h3>
              <p className="text-xs text-muted-foreground mt-1">
                {searchQuery ? "Try a different search term" : "Start a new chat to begin"}
              </p>
            </div>
          )}
        </div>
      </ScrollArea>

      <div className="border-t p-3">
        <Button variant="default" className="w-full" onClick={onNewChat}>
          <Plus className="mr-2 h-4 w-4" />
          New Chat
        </Button>
      </div>
    </div>
  )
}
