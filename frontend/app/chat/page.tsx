"use client"

import { useState, useRef, useEffect } from "react"
import { useSearchParams, useRouter } from "next/navigation"
import { Send, Mic, ChevronDown, Play, Pause, Maximize2, Clock, AlertTriangle, ArrowLeft, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { VideoModal } from "@/components/video-modal1"
import { ChatHistorySidebar, type ChatSession } from "@/components/chat-history-sidebar"
import { v4 as uuidv4 } from "uuid"

type MessageType = {
  id: string
  role: "user" | "ai"
  content: string
  timestamp: string
  videos?: {
    id: string
    title: string
    timestamp: string
    location: string
    url?: string
  }[]
  anomalies?: {
    type: string
    severity: "low" | "medium" | "high"
    description: string
  }[]
}

type ChatSessionData = {
  id: string
  title: string
  messages: MessageType[]
  createdAt: string
  updatedAt: string
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:5000/chat"


export default function ChatPage() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const initialQuery = searchParams.get("q") || ""

  const [chatSessions, setChatSessions] = useState<ChatSessionData[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string>("")
  const [input, setInput] = useState(initialQuery)
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [isPlaying, setIsPlaying] = useState<Record<string, boolean>>({})
  const [selectedVideo, setSelectedVideo] = useState<MessageType["videos"][0] | null>(null)
  const [isVideoModalOpen, setIsVideoModalOpen] = useState(false)
  const [showSidebar, setShowSidebar] = useState(true)

  // Load existing sessions from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem("chatSessions")
    if (saved) {
      const parsed: ChatSessionData[] = JSON.parse(saved)
      setChatSessions(parsed)
      setCurrentSessionId(parsed[0]?.id || "")
    } else {
      // Create initial empty session
      createNewSession()
    }
  }, [])

  // Persist sessions to localStorage
  useEffect(() => {
    if (chatSessions.length) {
      localStorage.setItem("chatSessions", JSON.stringify(chatSessions))
    }
  }, [chatSessions])

  // Auto scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [chatSessions, currentSessionId])

  // Fetch messages when session changes
  useEffect(() => {
    if (!currentSessionId) return
    fetch(`${API_BASE}/sessions/${currentSessionId}/messages`)
      .then((res) => res.json())
      .then((data) => {
        setChatSessions((prev) =>
          prev.map((session) =>
            session.id === currentSessionId
              ? { ...session, messages: data.messages as MessageType[] }
              : session,
          ),
        )
      })
  }, [currentSessionId])

  const createNewSession = async () => {
    // Create on backend
    const resp = await fetch(`${API_BASE}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input || undefined }),
    })
    const { session_id } = await resp.json()

    const newSession: ChatSessionData = {
      id: session_id,
      title: input ? input.substring(0, 30) : "New Chat",
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    }

    setChatSessions((prev) => [newSession, ...prev])
    setCurrentSessionId(session_id)
    setInput("")
  }

  const handleSelectSession = (sessionId: string) => {
    setCurrentSessionId(sessionId)
  }

  const handleDeleteSession = (sessionId: string) => {
    setChatSessions((prev) => prev.filter((session) => session.id !== sessionId))
    if (sessionId === currentSessionId) {
      const remaining = chatSessions.filter((s) => s.id !== sessionId)
      if (remaining.length) setCurrentSessionId(remaining[0].id)
      else createNewSession()
    }
  }

  const handleSendMessage = async (messageText: string = input) => {
    if (!messageText.trim()) return
    setIsTyping(true)

    // Optimistic add user message
    const userMsg: MessageType = {
      id: uuidv4(),
      role: "user",
      content: messageText,
      timestamp: new Date().toISOString(),
    }
    setChatSessions((prev) =>
      prev.map((session) =>
        session.id === currentSessionId
          ? { ...session, messages: [...session.messages, userMsg] }
          : session,
      ),
    )
    setInput("")

    // Send to backend
    const res = await fetch(
      `${API_BASE}/sessions/${currentSessionId}/messages`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: messageText }),
      },
    )
    const { aiResponse } = await res.json()

    // Append AI message
    setChatSessions((prev) =>
      prev.map((session) =>
        session.id === currentSessionId
          ? { ...session, messages: [...session.messages, aiResponse] }
          : session,
      ),
    )
    setIsTyping(false)
  }

  const toggleVideoPlayback = (videoId: string) => {
    setIsPlaying((prev) => ({ ...prev, [videoId]: !prev[videoId] }))
  }

  const openVideoModal = (video: MessageType["videos"][0]) => {
    setSelectedVideo(video)
    setIsVideoModalOpen(true)
  }

  const currentSession = chatSessions.find((s) => s.id === currentSessionId)
  const messages = currentSession?.messages || []

  const formattedChatSessions: ChatSession[] = chatSessions.map((session) => {
    if (!session.messages) return {
      id: session.id,
      title: session.title,
      preview: "Start a new conversation",
      timestamp: new Date(session.updatedAt),
      messageCount: 0,
    }
    const lastMessage = session.messages[session.messages.length - 1]
    console.log("lastMessage", lastMessage)
    console.log("session", session)
    return {
      id: session.id,
      title: session.title,
      preview:
        lastMessage?.content.substring(0, 60) +
        (lastMessage?.content.length > 60 ? "..." : ""),
      timestamp: new Date(session.updatedAt),
      messageCount: session.messages.length,
    }
  })

  return (
    <div className="flex h-full">
      {/* Chat History Sidebar */}
      {showSidebar && (
        <ChatHistorySidebar
          sessions={formattedChatSessions}
          currentSessionId={currentSessionId}
          onSelectSession={handleSelectSession}
          onNewChat={createNewSession}
          onDeleteSession={handleDeleteSession}
        />
      )}

      {/* Chat Interface */}
      <div className="flex flex-1 flex-col h-full">
        <div className="border-b p-3 sm:p-4 flex items-center justify-between">
          <div className="flex items-center gap-2 sm:gap-3">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowSidebar(!showSidebar)}
              className="md:flex"
              title={showSidebar ? "Hide chat history" : "Show chat history"}
            >
              <ArrowLeft className="h-5 w-5" />
            </Button>
            <div>
              <h1 className="text-lg sm:text-xl font-bold">AI Video Query Assistant</h1>
              <p className="text-xs sm:text-sm text-muted-foreground">
                Ask about any footage or events to find relevant video clips
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={createNewSession}>
              <Plus className="h-4 w-4 mr-2" />
              New Chat
            </Button>
            <Button variant="outline" size="sm" onClick={() => router.push("/cameras")}>
              View All Cameras
            </Button>
          </div>
        </div>

        <ScrollArea className="flex-1 p-4">
          <div className="mx-auto max-w-5xl space-y-6 pb-20">
            {messages.map((message) => (
              <div key={message.id} className="flex flex-col gap-3 animate-in fade-in-50 duration-300">
                <div className={message.role === "user" ? "chat-message-user" : "chat-message-ai"}>
                  {message.content}
                </div>

                {message.videos && message.videos.length > 0 && (
                  <div className="ml-4 mt-2">
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                      {message.videos.map((video) => (
                        <Card
                          key={video.id}
                          className="overflow-hidden hover:ring-2 hover:ring-primary/50 transition-all cursor-pointer"
                          onClick={() => openVideoModal(video)}
                        >
                            <div className="video-container relative aspect-video bg-black">
                            <video
                              src={video.url}
                              controls
                              className="h-full w-full object-cover"
                            />
                            <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between p-2 text-white">
                              <div className="flex items-center gap-2">
                              <Badge variant="outline" className="border-none bg-primary/20 text-primary-foreground">
                                {video.timestamp}
                              </Badge>
                              </div>
                              <div className="flex items-center gap-1">
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 text-white hover:bg-black/50"
                                onClick={(e) => {
                                e.stopPropagation()
                                toggleVideoPlayback(video.id)
                                }}
                              >
                                {isPlaying[video.id] ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 text-white hover:bg-black/50"
                                onClick={(e) => {
                                e.stopPropagation()
                                openVideoModal(video)
                                }}
                              >
                                <Maximize2 className="h-3 w-3" />
                              </Button>
                              </div>
                            </div>
                            </div>
                          <div className="p-2">
                            <h3 className="font-medium text-sm">{video.title}</h3>
                            <div className="mt-1 flex items-center text-xs text-muted-foreground">
                              <Clock className="mr-1 h-3 w-3" />
                              <span>{video.timestamp}</span>
                              <span className="mx-1">•</span>
                              <span>{video.location}</span>
                            </div>
                          </div>
                        </Card>
                      ))}
                    </div>
                  </div>
                )}

                {message.anomalies && message.anomalies.length > 0 && (
                  <div className="ml-4 mt-2 space-y-2">
                    <h4 className="text-sm font-medium">Detected Anomalies:</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                      {message.anomalies.map((anomaly, index) => (
                        <div
                          key={index}
                          className="flex items-start gap-2 rounded-md border p-2 hover:bg-muted/50 transition-colors"
                        >
                          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-red-100 text-red-700 dark:bg-red-900/20">
                            <AlertTriangle className="h-3 w-3" />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium">{anomaly.type}</span>
                              <Badge
                                variant={
                                  anomaly.severity === "high"
                                    ? "destructive"
                                    : anomaly.severity === "medium"
                                      ? "outline"
                                      : "secondary"
                                }
                                className="text-xs"
                              >
                                {anomaly.severity}
                              </Badge>
                            </div>
                            <p className="text-xs text-muted-foreground">{anomaly.description}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {isTyping && (
              <div className="chat-message-ai inline-block animate-in fade-in-50 duration-300">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        <div className="border-t bg-background p-4">
          <div className="mx-auto max-w-5xl flex items-center gap-2">
            <Button variant="outline" size="icon" className="shrink-0" title="Voice input">
              <Mic className="h-4 w-4" />
            </Button>
            <div className="relative flex-1">
              <Input
                className="pr-10 py-6"
                placeholder="Ask about any event or footage..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleSendMessage()
                  }
                }}
              />
              <Button
                variant="ghost"
                size="icon"
                className="absolute right-0 top-0 h-full"
                onClick={() => handleSendMessage()}
                disabled={!input.trim()}
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
            <Button variant="outline" size="icon" className="shrink-0">
              <ChevronDown className="h-4 w-4" />
            </Button>
          </div>
          <div className="mx-auto mt-3 max-w-5xl">
            <div className="flex flex-wrap gap-2">
              <Button variant="outline" size="sm" onClick={() => setInput("Show me all people entering after 11 PM")}>
                People after 11 PM
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setInput("Show suspicious activity in the parking lot")}
              >
                Suspicious activity
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setInput("Find all vehicles in the north parking lot")}
              >
                Vehicles in parking
              </Button>
              <Button variant="outline" size="sm" onClick={() => setInput("Show me unauthorized access attempts")}>
                Unauthorized access
              </Button>
            </div>
          </div>
        </div>

        {selectedVideo && (
          <VideoModal isOpen={isVideoModalOpen} onClose={() => setIsVideoModalOpen(false)} video={selectedVideo} />
        )}
      </div>
    </div>
  )
}
