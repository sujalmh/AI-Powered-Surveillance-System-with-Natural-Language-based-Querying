"use client";

import { useEffect, useState } from "react";
import { MessageSquare, Plus, Search } from "lucide-react";
import { api } from "@/lib/api";
import { ScrollArea } from "@/components/ui/scroll-area";

type SessionItem = {
  session_id: string;
  last_message?: string;
  last_message_time?: string;
  message_count: number;
};


export function ConversationSidebar() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [current, setCurrent] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    try {
      const data = await api.chatSessions();
      setSessions(Array.isArray(data) ? data : []);
      const sid = typeof window !== "undefined" ? window.localStorage.getItem("ai_surveillance_chat_session") : null;
      setCurrent(sid);
    } catch {}
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  const switchSession = (sid: string) => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("ai_surveillance_chat_session", sid);
      setCurrent(sid);
      window.dispatchEvent(new CustomEvent("chat:session:set", { detail: { session_id: sid } }));
    }
  };

  const newConversation = () => {
    const sid = crypto.randomUUID();
    switchSession(sid);
    setSessions((prev) => [
      { session_id: sid, last_message: "New conversation", last_message_time: new Date().toISOString(), message_count: 0 },
      ...prev,
    ]);
  };

  return (
    <div
      style={{
        width: "256px",
        maxWidth: "256px",
        flexShrink: 0,
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-sm)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        height: "100%",
        minWidth: 0,
        boxSizing: "border-box" as const,
      }}
    >
      {/* Header */}
      <div style={{
        height: "52px",
        borderBottom: "1px solid var(--color-border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 12px 0 14px",
        flexShrink: 0,
      }}>
        <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-text)" }}>Chats</span>
        <button
          onClick={newConversation}
          style={{
            width: "26px", height: "26px",
            borderRadius: "var(--radius-md)",
            background: "var(--color-primary-light)",
            border: "none",
            display: "flex", alignItems: "center", justifyContent: "center",
            color: "var(--color-primary)",
            cursor: "pointer",
            transition: "all 150ms",
          }}
          className="new-chat-btn"
          title="New conversation"
        >
          <Plus style={{ width: "14px", height: "14px" }} />
        </button>
      </div>

      {/* Search */}
      <div style={{ padding: "10px", flexShrink: 0 }}>
        <div style={{ position: "relative" }}>
          <Search style={{
            width: "12px", height: "12px",
            position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)",
            color: "var(--color-text-faint)", pointerEvents: "none",
          }} />
          <input
            type="text"
            placeholder="Search chats..."
            style={{
              width: "100%",
              background: "var(--color-surface-raised)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
              padding: "6px 10px 6px 26px",
              fontSize: "0.8125rem",
              color: "var(--color-text)",
              outline: "none",
              transition: "border-color 150ms",
              fontFamily: "var(--font-ui)",
            }}
            onFocus={e => (e.target.style.borderColor = "var(--color-primary)")}
            onBlur={e => (e.target.style.borderColor = "var(--color-border)")}
          />
        </div>
      </div>

      {/* Session list */}
      <ScrollArea className="sidebar-scroll-area" style={{ flex: 1, minHeight: 0, overflow: "hidden", width: "100%" }}>
        <div style={{ padding: "4px 0 4px 8px", display: "flex", flexDirection: "column", gap: "2px", width: "calc(256px - 2px - 8px)", boxSizing: "border-box" as const, overflow: "hidden" }}>
          {loading && sessions.length === 0 && Array.from({ length: 4 }).map((_, i) => (
            <div key={i} style={{
              height: "52px",
              background: "var(--color-surface-raised)",
              borderRadius: "var(--radius-md)",
              animation: "pulse-bg 1.5s ease-in-out infinite",
              marginBottom: "2px",
            }} />
          ))}
          {sessions.map((conv) => {
            const isActive = current === conv.session_id;

            return (
              <button
                key={conv.session_id}
                onClick={() => switchSession(conv.session_id)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  padding: "10px 12px",
                  borderRadius: "var(--radius-md)",
                  background: isActive ? "var(--color-primary-light)" : "transparent",
                  border: "none",
                  width: "100%",
                  minWidth: 0,
                  boxSizing: "border-box" as const,
                  overflow: "hidden",
                  cursor: "pointer",
                  textAlign: "left" as const,
                  transition: "background 150ms ease",
                }}
                className="session-item"
                data-active={isActive ? "true" : undefined}
              >
                <div style={{ flex: 1, minWidth: 0, overflow: "hidden" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: "6px", minWidth: 0 }}>
                    <span style={{
                      fontSize: "0.8125rem",
                      fontWeight: isActive ? 600 : 500,
                      color: isActive ? "var(--color-primary)" : "var(--color-text)",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      display: "block",
                      minWidth: 0,
                      flex: "1 1 0%",
                    }}>
                      {(() => { const t = conv.last_message || "New conversation"; return t.length > 32 ? t.slice(0, 32) + "…" : t; })()}
                    </span>
                    <span style={{ fontSize: "0.625rem", color: "var(--color-text-faint)", flexShrink: 0, whiteSpace: "nowrap" }}>
                      {conv.last_message_time ? new Date(conv.last_message_time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "Now"}
                    </span>
                  </div>
                </div>
              </button>
            );
          })}
          {sessions.length === 0 && !loading && (
            <div style={{
              textAlign: "center",
              padding: "32px 16px",
              color: "var(--color-text-faint)",
              fontSize: "0.8125rem",
            }}>
              No past sessions
            </div>
          )}
        </div>
      </ScrollArea>

      <style>{`
        .sidebar-scroll-area,
        .sidebar-scroll-area [data-slot="scroll-area-viewport"],
        .sidebar-scroll-area [data-slot="scroll-area-viewport"] > div {
          max-width: 100% !important;
          overflow-x: hidden !important;
        }
        .session-item:hover:not([data-active]) {
          background: var(--color-surface-raised) !important;
        }
        .new-chat-btn:hover {
          background: var(--color-primary) !important;
          color: white !important;
        }
        @keyframes pulse-bg { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
      `}</style>
    </div>
  );
}
