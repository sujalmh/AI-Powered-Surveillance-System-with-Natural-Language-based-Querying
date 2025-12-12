"use client";

import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";

type CameraDoc = {
  camera_id: number;
  source?: string;
  location?: string;
  status?: string;
  last_seen?: string;
  running?: boolean;
};

type AlertLog = {
  alert_id: string;
  triggered_at: string;
  camera_id?: number;
  message: string;
  snapshot?: string | null;
  clip?: string | null;
};

export function AlertsMap() {
  const [cameras, setCameras] = useState<CameraDoc[]>([]);
  const [logs, setLogs] = useState<AlertLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const [cams, lgs] = await Promise.all([api.listCameras(), api.listAlertLogs()]);
      setCameras(
        (cams as any[]).map((c) => ({
          camera_id: Number(c.camera_id),
          source: c.source,
          location: c.location,
          status: c.status,
          last_seen: c.last_seen,
          running: Boolean(c.running),
        }))
      );
      setLogs(lgs as any[]);
    } catch (e: any) {
      setErr(e?.message || "Failed to load map data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  // refresh when alerts change
  useEffect(() => {
    const handler = () => load();
    document.addEventListener("alerts:refresh", handler as any);
    return () => document.removeEventListener("alerts:refresh", handler as any);
  }, []);

  // Compute cameras with recent alerts (from the most recent 50 logs)
  const alertingCameras = useMemo(() => {
    const set = new Set<number>();
    for (const l of logs) {
      if (typeof l.camera_id === "number") set.add(l.camera_id);
    }
    return set;
  }, [logs]);

  return (
    <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6 h-full flex flex-col">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xl font-bold text-foreground">Camera Status</h2>
        {loading && <span className="text-xs text-muted-foreground">Loading…</span>}
        {err && <span className="text-xs text-red-400">{err}</span>}
      </div>

      <div className="flex-1 overflow-auto rounded-lg border border-white/10">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Camera</th>
              <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Location</th>
              <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Running</th>
              <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Alert</th>
              <th className="text-left py-2 px-3 text-sm font-semibold text-muted-foreground">Last Seen</th>
            </tr>
          </thead>
          <tbody>
            {cameras.map((c) => {
              const alerting = alertingCameras.has(c.camera_id);
              return (
                <tr key={c.camera_id} className="border-b border-border/50 hover:bg-white/5">
                  <td className="py-2 px-3 text-sm text-foreground font-medium">#{c.camera_id}</td>
                  <td className="py-2 px-3 text-sm text-muted-foreground">{c.location || "-"}</td>
                  <td className="py-2 px-3 text-sm">
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                        c.running
                          ? "bg-green-500/20 text-green-400 border border-green-500/50"
                          : "bg-gray-500/20 text-gray-400 border border-gray-500/50"
                      }`}
                    >
                      {c.running ? "Running" : "Stopped"}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-sm">
                    <span
                      className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                        alerting
                          ? "bg-red-500/20 text-red-400 border border-red-500/50"
                          : "bg-green-500/20 text-green-400 border border-green-500/50"
                      }`}
                    >
                      {alerting ? "Active Alert" : "Normal"}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-xs text-muted-foreground">
                    {c.last_seen ? new Date(c.last_seen).toLocaleString() : "-"}
                  </td>
                </tr>
              );
            })}
            {cameras.length === 0 && !loading && (
              <tr>
                <td colSpan={5} className="py-6 text-center text-sm text-muted-foreground">
                  No cameras registered yet.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="mt-4 space-y-2 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span className="text-muted-foreground">Active Alert</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span className="text-muted-foreground">Normal</span>
        </div>
      </div>
    </div>
  );
}
