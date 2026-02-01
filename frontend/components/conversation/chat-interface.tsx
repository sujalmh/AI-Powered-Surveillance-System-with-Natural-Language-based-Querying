"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { Zap, AlertCircle, ChevronDown } from "lucide-react"
import { ChatMessage } from "./chat-message"
import { QueryResults } from "./query-results"
import { api, type ChatSendResponse } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"

interface ChatInterfaceProps {
  onShowSteps: () => void
}

type UIMessage = { id: number; type: "user" | "assistant"; content: string; payload?: any }

function useChatSessionId() {
  const key = "ai_surveillance_chat_session"
  const [sid, setSid] = useState<string>("")
  // Initialize from localStorage once
  useEffect(() => {
    let v = typeof window !== "undefined" ? window.localStorage.getItem(key) : null
    if (!v) {
      v = crypto.randomUUID()
      if (typeof window !== "undefined") window.localStorage.setItem(key, v)
    }
    setSid(v || "")
  }, [])
  // Listen to "chat:session:set" events from the sidebar to switch sessions
  useEffect(() => {
    const handler = (e: Event) => {
      try {
        const ce = e as CustomEvent
        const next = ce?.detail?.session_id as string | undefined
        if (next && typeof next === "string") {
          setSid(next)
        }
      } catch {
        // no-op
      }
    }
    if (typeof window !== "undefined") {
      window.addEventListener("chat:session:set", handler as EventListener)
      return () => window.removeEventListener("chat:session:set", handler as EventListener)
    }
  }, [])
  return sid
}

export function ChatInterface({ onShowSteps }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<UIMessage[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [response, setResponse] = useState<ChatSendResponse | null>(null)
  const [mode, setMode] = useState<"retrieve" | "alert">("retrieve")
  const sessionId = useChatSessionId()
  const nextIdRef = useRef(2)
  
  // Scroll management
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const [showScrollButton, setShowScrollButton] = useState(false)
  const [isUserScrolling, setIsUserScrolling] = useState(false)

  const canSend = useMemo(() => !!input.trim() && !!sessionId && !loading, [input, sessionId, loading])

  // Load chat history for the current session id on mount / when session changes
  useEffect(() => {
    let mounted = true
    if (!sessionId) return
    ;(async () => {
      try {
        const hist = await api.chatHistory(sessionId)
        if (!mounted) return
        const msgs: UIMessage[] = (hist.messages || []).map((m, i) => ({
          id: i + 1,
          type: m.role === "user" ? "user" : "assistant",
          content: m.content || "",
          payload: (m as any).payload,
        }))
        setMessages(msgs)
        // set next id to continue after loaded history
        nextIdRef.current = msgs.length + 1
      } catch (e) {
        // ignore history load errors in UI; user can still chat
      }
    })()
    return () => {
      mounted = false
    }
  }, [sessionId])

  // Auto-scroll to bottom when messages change (unless user is scrolling)
  useEffect(() => {
    if (!isUserScrolling && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages, isUserScrolling])

  // Detect scroll position to show/hide scroll button
  useEffect(() => {
    const container = scrollContainerRef.current
    if (!container) return

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100
      
      setShowScrollButton(!isNearBottom)
      setIsUserScrolling(!isNearBottom)
    }

    container.addEventListener("scroll", handleScroll)
    return () => container.removeEventListener("scroll", handleScroll)
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
    setIsUserScrolling(false)
  }

  const handleSend = async () => {
    if (!canSend) return
    const text = input.trim()
    setInput("")
    setError(null)

    const userMsg: UIMessage = { id: nextIdRef.current++, type: "user", content: text }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      const resp = await api.chatSend(sessionId, text, 50)
      setResponse(resp)
      const assistantMsg: UIMessage = { id: nextIdRef.current++, type: "assistant", content: resp.answer, payload: resp }
      setMessages((prev) => [...prev, assistantMsg])
    } catch (e: any) {
      const msg = e?.message || "Request failed"
      setError(msg)
      const assistantMsg: UIMessage = {
        id: nextIdRef.current++,
        type: "assistant",
        content: `Error: ${msg}`,
      }
      setMessages((prev) => [...prev, assistantMsg])
    } finally {
      setLoading(false)
    }
  }

  const handleSetAlert = async () => {
    if (!canSend) return
    const text = input.trim()
    setInput("")
    setError(null)

    const userMsg: UIMessage = { id: nextIdRef.current++, type: "user", content: `[Alert Request] ${text}` }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      // For now, simulate alert creation with a message
      // In production, this would call an actual alert creation API
      const assistantMsg: UIMessage = {
        id: nextIdRef.current++,
        type: "assistant",
        content: `Alert created successfully! You will be notified when: "${text}"`,
      }
      setMessages((prev) => [...prev, assistantMsg])
    } catch (e: any) {
      const msg = e?.message || "Failed to create alert"
      setError(msg)
      const assistantMsg: UIMessage = {
        id: nextIdRef.current++,
        type: "assistant",
        content: `Error creating alert: ${msg}`,
      }
      setMessages((prev) => [...prev, assistantMsg])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex-1 glass-card glass-noise relative rounded-2xl p-6 flex flex-col overflow-hidden">
      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-accent to-transparent opacity-60" />
      
      <div ref={scrollContainerRef} className="flex-1 overflow-y-auto space-y-4 mb-6 relative z-10">
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} onShowSteps={onShowSteps} />
        ))}
        {/* Invisible marker for auto-scroll */}
        <div ref={messagesEndRef} />
      </div>

      {/* Floating scroll to bottom button */}
      {showScrollButton && (
        <button
          onClick={scrollToBottom}
          className="absolute bottom-24 right-8 z-20 p-3 rounded-full bg-primary/90 hover:bg-primary text-white shadow-lg transition-all duration-200 hover:scale-110"
          aria-label="Scroll to bottom"
        >
          <ChevronDown className="w-5 h-5" />
        </button>
      )}

      <div className="space-y-3 relative z-10">
        <div className="flex items-center gap-3">


          <div className="glass relative rounded-2xl p-4 group transition-all duration-300 hover:glass-glow flex-1">
            {/* Input accent indicator */}
            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-gradient-to-b from-accent to-primary rounded-r opacity-60 group-focus-within:opacity-100 transition-opacity" />
            
            <input
            type="text"
            placeholder="Ask: show people near Gate 3 in the last 30 minutes"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSend()
            }}
            disabled={!sessionId || loading}
            className="w-full bg-transparent text-foreground placeholder-muted-foreground outline-none disabled:opacity-60 pl-3"
          />
          </div>
        </div>

        {error && <p className="text-xs text-red-400 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
          {error}
        </p>}

        <div className="flex gap-3">
          <Button
            onClick={mode === "retrieve" ? handleSend : () => setMode("retrieve")}
            disabled={mode === "retrieve" && !canSend}
            variant={mode === "retrieve" ? "default" : "outline"}
            className="flex-1 rounded-xl h-auto py-3"
          >
            {loading && mode === "retrieve" ? <Spinner className="mr-2 text-white" /> : <Zap className="w-4 h-4" />}
            {mode === "retrieve" && loading ? "Retrieving..." : "Retrieve"}
          </Button>
          <Button
            onClick={mode === "alert" ? handleSetAlert : () => setMode("alert")}
            disabled={mode === "alert" && !canSend}
            variant={mode === "alert" ? "default" : "outline"}
            className="flex-1 rounded-xl h-auto py-3"
          >
            {loading && mode === "alert" ? <Spinner className="mr-2 text-white" /> : <AlertCircle className="w-4 h-4" />}
            {mode === "alert" && loading ? "Creating Alert..." : "Set Alert"}
          </Button>
        </div>
      </div>
    </div>
  )
}
