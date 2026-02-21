import React from "react"
import { Bot, User } from "lucide-react"
import { QueryResults } from "./query-results"
import type { ChatSendResponse } from "@/lib/api"

interface ChatMessageProps {
  message: {
    id: number
    type: "user" | "assistant"
    content: string
    payload?: any
  }
  onShowSteps?: () => void
}

export const ChatMessage = React.memo(({ message, onShowSteps }: ChatMessageProps) => {
  const resp = (message?.payload || null) as ChatSendResponse | null
  const hasMedia =
    !!resp &&
    (((resp.combined_tracks?.length || 0) > 0) ||
     ((resp.merged_tracks?.length || 0) > 0) ||
     ((resp.results?.length || 0) > 0))

  const isUser = message.type === "user"

  return (
    <div style={{ display: "flex", gap: "10px", justifyContent: isUser ? "flex-end" : "flex-start" }}>
      {/* Avatar — assistant only */}
      {!isUser && (
        <div style={{
          width: "28px", height: "28px",
          borderRadius: "var(--radius-full)",
          background: "var(--color-surface-raised)",
          border: "1px solid var(--color-border)",
          display: "flex", alignItems: "center", justifyContent: "center",
          flexShrink: 0,
          marginTop: "2px",
        }}>
          <Bot style={{ width: "13px", height: "13px", color: "var(--color-text-muted)" }} />
        </div>
      )}

      {/* Bubble */}
      <div style={{
        maxWidth: "72%",
        background: isUser
          ? "linear-gradient(135deg, #15803D 0%, #16A34A 100%)"
          : "var(--color-surface-raised)",
        border: isUser
          ? "none"
          : "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        borderTopLeftRadius: isUser ? "var(--radius-lg)" : "4px",
        borderTopRightRadius: isUser ? "4px" : "var(--radius-lg)",
        padding: "10px 14px",
        boxShadow: isUser ? "var(--shadow-sm)" : "var(--shadow-xs)",
      }}>
        <p style={{
          fontSize: "0.875rem",
          color: isUser ? "white" : "var(--color-text)",
          lineHeight: 1.55,
          whiteSpace: "pre-wrap",
          margin: 0,
        }}>
          {message.content}
        </p>
        {!isUser && hasMedia && (
          <div style={{ marginTop: "12px" }}>
            <QueryResults onShowSteps={onShowSteps || (() => {})} response={resp} />
          </div>
        )}
      </div>

      {/* Avatar — user only */}
      {isUser && (
        <div style={{
          width: "28px", height: "28px",
          borderRadius: "var(--radius-full)",
          background: "linear-gradient(135deg, #15803D 0%, #16A34A 100%)",
          display: "flex", alignItems: "center", justifyContent: "center",
          flexShrink: 0,
          marginTop: "2px",
        }}>
          <User style={{ width: "13px", height: "13px", color: "white" }} />
        </div>
      )}
    </div>
  )
})
ChatMessage.displayName = "ChatMessage"
