"use client";

import { useEffect, useState } from "react";
import { MessageSquare, Plus } from "lucide-react";
import { api } from "@/lib/api";

type SessionItem = {
  session_id: string;
  last_message?: string;
  last_message_time?: string;
  message_count: number;
};

export function ConversationSidebar() {
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [current, setCurrent] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const data = await api.chatSessions();
      setSessions(Array.isArray(data) ? data : []);
      const sid = typeof window !== "undefined" ? window.localStorage.getItem("ai_surveillance_chat_session") : null;
      setCurrent(sid);
    } catch (e: any) {
      setErr(e?.message || "Failed to load sessions");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const switchSession = (sid: string) => {
    try {
      if (typeof window !== "undefined") {
        window.localStorage.setItem("ai_surveillance_chat_session", sid);
        setCurrent(sid);
        // Notify listeners (ChatInterface) to reload history
        window.dispatchEvent(new CustomEvent("chat:session:set", { detail: { session_id: sid } }));
      }
    } catch {
      // ignore
    }
  };

  const newConversation = () => {
    const sid = crypto.randomUUID();
    switchSession(sid);
    // Optimistically add to list top
    setSessions((prev) => [
      { session_id: sid, last_message: "New conversation", last_message_time: new Date().toISOString(), message_count: 0 },
      ...prev,
    ]);
  };

  return (
    <div className="w-64 glass-card glass-noise relative rounded-2xl p-4 flex flex-col overflow-hidden">
      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-accent to-transparent opacity-60" />
      
      <button onClick={newConversation} className="gradient-button w-full mb-4 py-3 rounded-xl font-semibold flex items-center justify-center gap-2 relative group overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />
        <Plus className="w-4 h-4 relative z-10" />
        <span className="relative z-10">New Conversation</span>
      </button>

      <div className="flex items-center justify-between mb-2 relative z-10">
        <p className="text-xs text-muted-foreground flex items-center gap-2">
          Conversations
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
        </p>
        <button
          onClick={load}
          className="px-2 py-1 rounded glass text-[10px] text-foreground hover:glass-glow transition-all duration-300"
        >
          Refresh
        </button>
      </div>
      {err && <div className="text-xs text-red-400 mb-2">{err}</div>}
      {loading && <div className="text-xs text-muted-foreground mb-2">Loading…</div>}

      <div className="flex-1 overflow-y-auto space-y-2 relative z-10">
        {sessions.map((conv) => {
          const isActive = current === conv.session_id;
          return (
            <button
              key={conv.session_id}
              onClick={() => switchSession(conv.session_id)}
              className={`w-full text-left glass relative rounded-2xl p-3 group transition-all duration-300 hover:glass-glow ${
                isActive ? "gradient-button" : ""
              }`}
              title={conv.session_id}
            >
              {/* Active conversation accent */}
              {isActive && (
                <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-gradient-to-b from-accent to-primary rounded-r" />
              )}
              
              <div className="flex items-start gap-2 relative z-10">
                <div className={`p-1.5 rounded-lg transition-all duration-300 ${
                  isActive ? "bg-primary-foreground/10" : "bg-accent/10"
                } group-hover:scale-110`}>
                  <MessageSquare className={`w-4 h-4 ${isActive ? "text-primary-foreground" : "text-accent"}`} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className={`text-sm font-medium truncate ${
                    isActive ? "text-primary-foreground" : "text-foreground"
                  }`}>
                    {conv.last_message || "Empty conversation"}
                  </p>
                  <p className={`text-[10px] ${
                    isActive ? "text-primary-foreground/70" : "text-muted-foreground"
                  }`}>
                    {conv.message_count} message{conv.message_count === 1 ? "" : "s"}
                    {conv.last_message_time ? ` · ${new Date(conv.last_message_time).toLocaleString()}` : ""}
                  </p>
                </div>
              </div>
            </button>
          );
        })}
        {sessions.length === 0 && !loading && (
          <div className="text-xs text-muted-foreground">No conversations yet. Create a new one.</div>
        )}
      </div>
    </div>
  );
}
