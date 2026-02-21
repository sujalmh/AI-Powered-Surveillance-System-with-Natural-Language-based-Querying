"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { Zap, AlertCircle, Send, ChevronLeft, ChevronRight } from "lucide-react"
import { ChatMessage } from "./chat-message"
import { api, type ChatSendResponse } from "@/lib/api"

interface ChatInterfaceProps {
  onShowSteps: () => void
}

type UIMessage = { id: number; type: "user" | "assistant"; content: string; payload?: any }

function useChatSessionId() {
  const key = "ai_surveillance_chat_session"
  const [sid, setSid] = useState<string>("")
  useEffect(() => {
    let v = typeof window !== "undefined" ? window.localStorage.getItem(key) : null
    if (!v) { v = crypto.randomUUID(); if (typeof window !== "undefined") window.localStorage.setItem(key, v) }
    setSid(v || "")
  }, [])
  useEffect(() => {
    const handler = (e: Event) => {
      try { const ce = e as CustomEvent; const next = ce?.detail?.session_id as string | undefined; if (next) setSid(next) } catch {}
    }
    if (typeof window !== "undefined") { window.addEventListener("chat:session:set", handler as EventListener); return () => window.removeEventListener("chat:session:set", handler as EventListener) }
  }, [])
  return sid
}

export function ChatInterface({ onShowSteps }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<UIMessage[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<"retrieve" | "alert">("retrieve")
  const [showDetails, setShowDetails] = useState(false)

  const sessionId = useChatSessionId()
  const nextIdRef = useRef(2)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const canSend = useMemo(() => !!input.trim() && !!sessionId && !loading, [input, sessionId, loading])

  useEffect(() => {
    let mounted = true
    if (!sessionId) return
    ;(async () => {
      try {
        const hist = await api.chatHistory(sessionId)
        if (!mounted) return
        const msgs: UIMessage[] = (hist.messages || []).map((m: any, i: number) => ({
          id: i + 1, type: m.role === "user" ? "user" : "assistant", content: m.content || "", payload: m.payload,
        }))
        setMessages(msgs)
        nextIdRef.current = msgs.length + 1
      } catch {}
    })()
    return () => { mounted = false }
  }, [sessionId])

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }) }, [messages])

  const sendMessage = async (text: string, isAlert: boolean) => {
    const userMsg: UIMessage = { id: nextIdRef.current++, type: "user", content: isAlert ? `[Alert] ${text}` : text }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true); setError(null)
    try {
      if (isAlert) {
        setMessages((prev) => [...prev, { id: nextIdRef.current++, type: "assistant", content: `Alert created: "${text}"` }])
      } else {
        const resp = await api.chatSend(sessionId, text, 50)
        setMessages((prev) => [...prev, { id: nextIdRef.current++, type: "assistant", content: resp.answer, payload: resp }])
      }
    } catch (e: any) {
      const msg = e?.message || "Request failed"; setError(msg)
      setMessages((prev) => [...prev, { id: nextIdRef.current++, type: "assistant", content: `Error: ${msg}` }])
    } finally { setLoading(false) }
  }

  const submit = () => {
    if (!canSend) return
    const text = input.trim(); setInput(""); sendMessage(text, mode === "alert")
  }

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", background: "var(--color-surface)", overflow: "hidden" }}>

      {/* ── Top bar ── */}
      <div style={{
        height: "52px",
        borderBottom: "1px solid var(--color-border)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "0 16px", flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <Zap style={{ width: "14px", height: "14px", color: "var(--color-primary)" }} />
          <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-text)" }}>Current Session</span>
        </div>

        {/* More Details toggle */}
        <button
          onClick={() => { setShowDetails(v => !v); onShowSteps() }}
          style={{
            display: "flex", alignItems: "center", gap: "6px",
            padding: "5px 12px",
            borderRadius: "var(--radius-md)",
            background: showDetails ? "var(--color-primary-light)" : "var(--color-surface-raised)",
            border: `1px solid ${showDetails ? "rgba(22,163,74,0.3)" : "var(--color-border)"}`,
            color: showDetails ? "var(--color-primary)" : "var(--color-text-muted)",
            fontSize: "0.8125rem", fontWeight: 500,
            cursor: "pointer", transition: "all 150ms ease",
            fontFamily: "var(--font-ui)",
          }}
          className="details-btn"
        >
          {showDetails
            ? <ChevronRight style={{ width: "13px", height: "13px" }} />
            : <ChevronLeft style={{ width: "13px", height: "13px" }} />}
          Details
        </button>
      </div>

      {/* ── Messages ── */}
      {/* outer scroll container */}
      <div style={{ flex: 1, overflowY: "auto", display: "flex", flexDirection: "column" }}>
        {/* spacer pushes content to bottom */}
        <div style={{ flex: 1 }} />
        <div style={{ padding: "12px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
            {messages.length === 0 && !loading && (
              <div style={{
                display: "flex", flexDirection: "column", alignItems: "center",
                justifyContent: "center", padding: "40px 0", gap: "12px",
                color: "var(--color-text-faint)",
              }}>
                <div style={{
                  width: "44px", height: "44px", borderRadius: "var(--radius-full)",
                  background: "var(--color-primary-light)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}>
                  <Zap style={{ width: "20px", height: "20px", color: "var(--color-primary)" }} />
                </div>
                <p style={{ fontSize: "0.875rem" }}>Ask about anything in your surveillance footage</p>
              </div>
            )}
            {messages.map((msg) => <ChatMessage key={msg.id} message={msg} onShowSteps={onShowSteps} />)}
            {loading && (
              <div style={{ display: "flex", justifyContent: "flex-start" }}>
                <div style={{
                  background: "var(--color-surface-raised)", border: "1px solid var(--color-border)",
                  borderRadius: "var(--radius-lg)", borderTopLeftRadius: "4px",
                  padding: "10px 14px", display: "flex", alignItems: "center", gap: "8px",
                  boxShadow: "var(--shadow-xs)",
                }}>
                  <div style={{
                    width: "16px", height: "16px",
                    border: "2px solid var(--color-border)", borderTopColor: "var(--color-primary)",
                    borderRadius: "50%", animation: "spin 1s linear infinite",
                  }} />
                  <span style={{ fontSize: "0.8125rem", color: "var(--color-text-muted)" }}>Processing...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      {/* ── Input ── */}
      <div style={{
        padding: "10px 16px 12px",
        borderTop: "1px solid var(--color-border)",
        background: "var(--color-surface)",
        flexShrink: 0,
      }}>
        {error && (
          <p style={{ fontSize: "0.75rem", color: "var(--color-danger)", marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
            <AlertCircle style={{ width: "12px", height: "12px" }} /> {error}
          </p>
        )}
        <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
          {/* Input island */}
          <div style={{
            flex: 1, display: "flex", alignItems: "center", gap: "6px",
            background: "var(--color-surface-raised)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-xl)",
            boxShadow: "var(--shadow-md)",
            padding: "4px",
            transition: "border-color 150ms, box-shadow 150ms",
          }}
            onFocusCapture={e => {
              const el = e.currentTarget as HTMLElement
              el.style.borderColor = "var(--color-primary)"
              el.style.boxShadow = `var(--shadow-md), 0 0 0 3px var(--color-primary-light)`
            }}
            onBlurCapture={e => {
              const el = e.currentTarget as HTMLElement
              el.style.borderColor = "var(--color-border)"
              el.style.boxShadow = "var(--shadow-md)"
            }}
          >
            <input
              type="text"
              placeholder={mode === "retrieve" ? "Find people near Gate 3 in the last 30 minutes..." : "Alert me when a red car arrives..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              style={{
                flex: 1, height: "36px",
                background: "transparent", border: "none", outline: "none",
                padding: "0 10px",
                fontSize: "0.875rem", color: "var(--color-text)",
                fontFamily: "var(--font-ui)",
              }}
            />
            <button
              disabled={!canSend}
              onClick={submit}
              style={{
                width: "32px", height: "32px",
                borderRadius: "var(--radius-md)",
                background: canSend ? "linear-gradient(135deg, #15803D 0%, #16A34A 100%)" : "var(--color-border)",
                border: "none", display: "flex", alignItems: "center", justifyContent: "center",
                color: "white", cursor: canSend ? "pointer" : "not-allowed",
                transition: "all 150ms ease", boxShadow: canSend ? "var(--shadow-sm)" : "none",
                flexShrink: 0,
              }}
              className="send-btn"
            >
              <Send style={{ width: "13px", height: "13px" }} />
            </button>
          </div>

          {/* Mode toggle — outside input, to the right */}
          <div style={{
            display: "flex", alignItems: "center",
            background: "var(--color-surface-raised)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-xl)",
            boxShadow: "var(--shadow-md)",
            padding: "3px",
            flexShrink: 0,
          }}>
            <button
              onClick={() => setMode("retrieve")}
              style={{
                padding: "5px 12px", borderRadius: "var(--radius-lg)",
                fontSize: "0.75rem", fontWeight: 600,
                background: mode === "retrieve" ? "linear-gradient(135deg,#15803D,#16A34A)" : "transparent",
                color: mode === "retrieve" ? "white" : "var(--color-text-muted)",
                border: "none", cursor: "pointer", transition: "all 150ms ease",
                display: "flex", alignItems: "center", gap: "4px",
                fontFamily: "var(--font-ui)",
              }}
            >
              <Zap style={{ width: "10px", height: "10px" }} />
              Search
            </button>
            <button
              onClick={() => setMode("alert")}
              style={{
                padding: "5px 12px", borderRadius: "var(--radius-lg)",
                fontSize: "0.75rem", fontWeight: 600,
                background: mode === "alert" ? "#DC2626" : "transparent",
                color: mode === "alert" ? "white" : "var(--color-text-muted)",
                border: "none", cursor: "pointer", transition: "all 150ms ease",
                display: "flex", alignItems: "center", gap: "4px",
                fontFamily: "var(--font-ui)",
              }}
            >
              <AlertCircle style={{ width: "10px", height: "10px" }} />
              Alert
            </button>
          </div>
        </div>
        <p style={{ fontSize: "0.6875rem", color: "var(--color-text-faint)", textAlign: "center", marginTop: "5px" }}>
            Press Enter to send
          </p>
      </div>

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        .send-btn:hover:not([disabled]) { transform: scale(1.08); box-shadow: var(--shadow-md) !important; }
        .send-btn:active:not([disabled]) { transform: scale(0.96); }
        .details-btn:hover { background: var(--color-primary-light) !important; color: var(--color-primary) !important; }
      `}</style>
    </div>
  )
}
