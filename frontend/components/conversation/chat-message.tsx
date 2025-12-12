import { User, Bot } from "lucide-react"
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

export function ChatMessage({ message, onShowSteps }: ChatMessageProps) {
  const resp = (message?.payload || null) as ChatSendResponse | null
  const hasMedia =
    !!resp &&
    (((resp.combined_tracks?.length || 0) > 0) ||
     ((resp.merged_tracks?.length || 0) > 0) ||
     ((resp.results?.length || 0) > 0))

  return (
    <div className={`flex gap-4 ${message.type === "user" ? "justify-end" : "justify-start"} group`}>
      {message.type === "assistant" && (
        <div className="w-8 h-8 rounded-lg gradient-button flex items-center justify-center flex-shrink-0 shadow-md group-hover:shadow-lg group-hover:scale-110 transition-all duration-300 relative">
          <Bot className="w-5 h-5 text-primary-foreground" />
          <div className="absolute inset-0 bg-accent/20 rounded-lg blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10" />
        </div>
      )}

      <div
        className={`max-w-md relative rounded-2xl p-4 border transition-colors group-hover:glass-glow ${
          message.type === "user"
            ? "bg-[#3B82F6]/10 border-[#3B82F6]/25 text-[color:var(--foreground)]"
            : "bg-[color-mix(in oklab,var(--foreground) 4%,transparent)] border-[color:var(--border)]"
        }`}
      >
        {/* Message accent line */}
        <div className={`absolute top-0 left-0 right-0 h-[2px] rounded-t-2xl ${
          message.type === "user" 
            ? "bg-gradient-to-r from-transparent via-primary to-transparent opacity-60" 
            : "bg-gradient-to-r from-transparent via-accent to-transparent opacity-60"
        }`} />
        
        <p className="text-sm text-foreground relative z-10">{message.content}</p>
        {message.type === "assistant" && hasMedia && (
          <div className="mt-3 -mx-2 relative z-10">
            <QueryResults onShowSteps={onShowSteps || (() => {})} response={resp} />
          </div>
        )}
      </div>

      {message.type === "user" && (
        <div className="w-8 h-8 rounded-lg gradient-button flex items-center justify-center flex-shrink-0 shadow-md group-hover:shadow-lg group-hover:scale-110 transition-all duration-300 relative">
          <User className="w-5 h-5 text-primary-foreground" />
          <div className="absolute inset-0 bg-primary/20 rounded-lg blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10" />
        </div>
      )}
    </div>
  )
}
