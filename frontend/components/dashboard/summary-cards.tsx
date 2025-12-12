"use client";

import { Camera, AlertTriangle, TrendingUp, Clock } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";

export function SummaryCards() {
  const [activeCameras, setActiveCameras] = useState<number>(0);
  const [liveAlerts, setLiveAlerts] = useState<number>(0);
  const [anomalies, setAnomalies] = useState<number>(0);
  const [avgResponse, setAvgResponse] = useState<string>("-");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      // Active cameras
      const cams = (await api.listCameras()) as any[];
      setActiveCameras(cams.filter((c) => c.running).length);

      // Live alerts: consider alert logs in last 15 minutes as "live"
      const logs = (await api.listAlertLogs()) as any[];
      const now = Date.now();
      const live = logs.filter((l) => {
        const t = new Date(l.triggered_at).getTime();
        return now - t <= 15 * 60 * 1000;
      }).length;
      setLiveAlerts(live);

      // Anomalies today: count detections of objects (e.g., person) today; simplistic
      const start = new Date();
      start.setHours(0, 0, 0, 0);
      const params = new URLSearchParams();
      params.set("from", start.toISOString());
      const dets = (await api.listDetections(params)) as any[];
      const anomalyCount = dets.reduce((acc, d) => acc + (Array.isArray(d.objects) ? d.objects.length : 0), 0);
      setAnomalies(anomalyCount);

      // Avg. response time: not available yet -> compute mock from logs (diff between now and triggered)
      if (logs.length > 0) {
        const diffs = logs.slice(0, 10).map((l) => Math.max(1, Math.floor((now - new Date(l.triggered_at).getTime()) / 1000)));
        const avg = Math.floor(diffs.reduce((a, b) => a + b, 0) / diffs.length);
        setAvgResponse(`${avg}s`);
      } else {
        setAvgResponse("-");
      }
    } catch (e: any) {
      setErr(e?.message || "Failed to load summary");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const cards = useMemo(
    () => [
      {
        title: "Active Cameras",
        value: String(activeCameras),
        icon: Camera,
      },
      {
        title: "Live Alerts",
        value: String(liveAlerts),
        icon: AlertTriangle,
      },
      {
        title: "Anomalies Today",
        value: String(anomalies),
        icon: TrendingUp,
      },
      {
        title: "Avg. Response Time",
        value: avgResponse,
        icon: Clock,
      },
    ],
    [activeCameras, liveAlerts, anomalies, avgResponse]
  );

  const colorByTitle: Record<string, string> = {
    "Active Cameras": "#3B82F6",     // blue-500
    "Live Alerts": "#F59E0B",        // amber-500
    "Anomalies Today": "#EF4444",    // red-500
    "Avg. Response Time": "#06B6D4", // cyan-500
  };

  return (
    <div className="grid grid-cols-4 gap-4">
      {cards.map((card) => {
        const Icon = card.icon;
        const col = colorByTitle[card.title] || "var(--accent)";
        return (
          <div
            key={card.title}
            className="glass-card glass-noise relative rounded-2xl p-4 transition-all duration-300 hover:glass-glow group overflow-hidden"
          >
            {/* Colored left border */}
            <div
              className="absolute left-0 top-0 bottom-0 w-1 rounded-l-2xl opacity-60"
              style={{ background: `linear-gradient(to bottom, ${col}, transparent)` }}
            />

            {/* Accent border glow */}
            <div className="absolute -inset-[1px] rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"
                 style={{ background: `linear-gradient(to bottom right, ${col}33, transparent)` }} />

            {/* Top accent line */}
            <div className="absolute top-0 left-0 right-0 h-[2px]" style={{ background: `linear-gradient(to right, transparent, ${col}, transparent)` }} />

            <div className="flex items-start justify-between mb-4 relative z-10">
              <div>
                <p className="text-sm text-muted-foreground mb-1 flex items-center gap-2">
                  {card.title}
                  <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ backgroundColor: col }} />
                </p>
                <p className="text-3xl font-extrabold text-card-foreground tracking-tight">{card.value}</p>
              </div>
              <div className={cn("p-3 rounded-full shadow-sm relative group-hover:shadow-lg group-hover:scale-110 transition-all duration-300",)}
                   style={{ backgroundColor: `${col}26`, color: col }}>
                <Icon className="w-6 h-6" />
                {/* Icon glow effect */}
                <div className="absolute inset-0 rounded-full blur-md opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10"
                     style={{ backgroundColor: `${col}33` }} />
              </div>
            </div>
            <div className="h-1 bg-border rounded-full overflow-hidden backdrop-blur-sm relative">
              <div className="h-full w-3/4 transition-all duration-300 shadow-sm relative z-10"
                   style={{ background: `linear-gradient(to right, ${col}, var(--primary))` }}>
                {/* Progress bar shimmer */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent to-transparent animate-shimmer"
                     style={{ backgroundImage: `linear-gradient(90deg, transparent 0%, ${col}33 50%, transparent 100%)` }} />
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
