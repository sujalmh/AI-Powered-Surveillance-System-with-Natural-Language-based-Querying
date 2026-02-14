"use client";

import { AlertCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

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
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs} hour${hrs === 1 ? "" : "s"} ago`;
  const days = Math.floor(hrs / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

export function RecentAlerts() {
  const [logs, setLogs] = useState<AlertLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const res = (await api.listAlertLogs()) as any[];
      const list: AlertLog[] = res.map((l) => ({
        id: String(l.id),
        alert_id: String(l.alert_id),
        triggered_at: l.triggered_at,
        camera_id: typeof l.camera_id === "number" ? l.camera_id : undefined,
        message: l.message || "Alert triggered",
      }));
      setLogs(list);
      return list;
    } catch (e: any) {
      setErr(e?.message || "Failed to load alerts");
      return [];
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let evtSource: EventSource | null = null;
    let aborted = false;

    // Initial load then stream
    load().then((initialLogs) => {
      if (aborted) return;

      // Find latest timestamp to resume stream from
      let lastTs = undefined;
      if (initialLogs.length > 0) {
        // logs are sorted triggered_at desc, so first is latest
        lastTs = initialLogs[0].triggered_at;
      }

      // Start SSE
      evtSource = api.streamAlerts({ last_ts: lastTs });
      evtSource.onmessage = (e) => {
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

          setLogs((prev) => {
            // Avoid duplicates
            if (prev.some((p) => p.id === newLog.id)) return prev;
            return [newLog, ...prev];
          });
        } catch (err) {
          console.error("Failed to parse SSE alert event", err);
        }
      };

      evtSource.onerror = (e) => {
        if (aborted) return;
        // Check if connection closed or error
        if (evtSource?.readyState === EventSource.CLOSED) {
          // optional: reconnect logic is handled by browser usually, but customized retry can go here
        }
      };
    });

    return () => {
      aborted = true;
      if (evtSource) {
        evtSource.close();
      }
    };
  }, []);

  return (
    <div className="bg-card border border-border rounded-2xl p-4 h-full shadow-[0_2px_10px_rgba(25,24,59,0.1)]">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-card-foreground">Recent Alerts</h2>
        {loading && <span className="text-xs text-muted-foreground">Loading…</span>}
        {err && <span className="text-xs text-destructive">{err}</span>}
        <button
          onClick={() => load()}
          className="ml-auto px-3 py-1.5 rounded-lg bg-accent/10 border border-accent text-xs text-accent-foreground hover:bg-accent/20 transition-colors"
        >
          Refresh
        </button>
      </div>
      <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2">
        {logs.map((alert) => (
          <div
            key={alert.id}
            className="bg-muted/20 border border-border rounded-2xl p-3 shadow-[0_2px_8px_rgba(25,24,59,0.08)] hover:shadow-[0_0_15px_rgba(161,194,189,0.2)] transition-all duration-300 animate-in fade-in slide-in-from-top-2"
          >
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-accent" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-card-foreground">{alert.message}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  {timeAgo(alert.triggered_at)}
                  {typeof alert.camera_id === "number" ? ` · camera #${alert.camera_id}` : ""}
                </p>
              </div>
            </div>
          </div>
        ))}
        {logs.length === 0 && !loading && (
          <div className="text-sm text-muted-foreground">No recent alerts.</div>
        )}
      </div>
    </div>
  );
}
