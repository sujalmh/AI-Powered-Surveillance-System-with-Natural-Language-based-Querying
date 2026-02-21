"use client";

import { AlertTriangle } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { ScrollArea } from "@/components/ui/scroll-area";

type AlertLog = {
  id: string;
  alert_id: string;
  triggered_at: string;
  camera_id?: number;
  message: string;
};

function timeAgo(ts: string): string {
  const d = new Date(ts);
  const diffMs = Date.now() - d.getTime();
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

export function RecentAlerts() {
  const [logs, setLogs] = useState<AlertLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let evtSource: EventSource | null = null;
    let aborted = false;

    (async () => {
      setLoading(true);
      try {
        const res = (await api.listAlertLogs()) as any[];
        if (!aborted) {
          const list: AlertLog[] = res.map((l) => ({
            id: String(l.id),
            alert_id: String(l.alert_id),
            triggered_at: l.triggered_at,
            camera_id: typeof l.camera_id === "number" ? l.camera_id : undefined,
            message: l.message || "Alert triggered",
          }));
          setLogs(list);
          setLoading(false);

          evtSource = api.streamAlerts({ last_ts: list[0]?.triggered_at });
          evtSource.onmessage = (e: MessageEvent) => {
            if (aborted) return;
            try {
              const data = JSON.parse(e.data);
              const newLog: AlertLog = {
                id: String(data.id),
                alert_id: String(data.alert_id),
                triggered_at: data.triggered_at,
                camera_id: typeof data.camera_id === "number" ? data.camera_id : undefined,
                message: data.message || "Alert triggered",
              };
              setLogs((prev) => prev.some((p) => p.id === newLog.id) ? prev : [newLog, ...prev]);
            } catch {}
          };
        }
      } catch (e: any) {
        if (!aborted) {
          setErr(e?.message || "Failed to load");
          setLoading(false);
        }
      }
    })();

    return () => {
      aborted = true;
      if (evtSource) evtSource.close();
    };
  }, []);

  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-sm)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        width: "280px",
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div style={{
        padding: "14px 16px",
        borderBottom: "1px solid var(--color-border)",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <p style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--color-text)" }}>Recent Alerts</p>
        {loading && (
          <span style={{
            width: "6px", height: "6px",
            borderRadius: "50%",
            background: "var(--color-primary)",
            display: "inline-block",
            animation: "pulse 2s infinite",
          }} />
        )}
      </div>

      {/* Body */}
      <ScrollArea style={{ flex: 1, maxHeight: "400px" }}>
        <div style={{ padding: "10px", display: "flex", flexDirection: "column", gap: "6px" }}>
          {err && (
            <p style={{ fontSize: "0.75rem", color: "var(--color-danger)", padding: "8px" }}>{err}</p>
          )}
          {logs.length === 0 && !loading && (
            <div style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "32px 16px",
              gap: "8px",
              color: "var(--color-text-faint)",
            }}>
              <AlertTriangle style={{ width: "24px", height: "24px", opacity: 0.4 }} />
              <p style={{ fontSize: "0.8125rem", textAlign: "center" }}>No recent alerts</p>
            </div>
          )}
          {loading && logs.length === 0 && Array.from({ length: 4 }).map((_, i) => (
            <div key={i} style={{
              height: "56px",
              background: "var(--color-surface-raised)",
              borderRadius: "var(--radius-md)",
              animation: "pulse-bg 1.5s ease-in-out infinite",
            }} />
          ))}
          {logs.map((alert) => (
            <div key={alert.id} style={{
              background: "var(--color-surface-raised)",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)",
              padding: "10px 12px",
              display: "flex",
              alignItems: "flex-start",
              gap: "10px",
              transition: "background 150ms ease",
            }} className="alert-row">
              <AlertTriangle style={{
                width: "14px", height: "14px",
                color: "#D97706",
                flexShrink: 0,
                marginTop: "2px",
              }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{
                  fontSize: "0.8125rem",
                  color: "var(--color-text)",
                  fontWeight: 500,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  wordBreak: "break-word",
                  display: "-webkit-box",
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: "vertical",
                }}>
                  {alert.message}
                </p>
                <p style={{ fontSize: "0.6875rem", color: "var(--color-text-faint)", marginTop: "2px" }}>
                  {timeAgo(alert.triggered_at)}
                  {typeof alert.camera_id === "number" && (
                    <>
                      {" · "}
                      <span style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-muted)" }}>
                        #{alert.camera_id}
                      </span>
                    </>
                  )}
                </p>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>

      <style>{`
        .alert-row:hover { background: var(--color-bg) !important; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
        @keyframes pulse-bg { 0%,100% { opacity:1; } 50% { opacity:0.5; } }
      `}</style>
    </div>
  );
}
