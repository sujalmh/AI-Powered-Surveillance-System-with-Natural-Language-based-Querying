"use client";

import { Camera, AlertTriangle, TrendingUp, Activity, ArrowUpRight } from "lucide-react";
import { useMemo } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";

const fetchSummaryStats = async () => {
  const [cams, logsRes, detsRes] = await Promise.allSettled([
    api.listCameras(),
    api.listAlertLogs(),
    (async () => {
      const start = new Date();
      start.setHours(0, 0, 0, 0);
      const detParams = new URLSearchParams();
      detParams.set("from", start.toISOString());
      return api.listDetections(detParams);
    })(),
  ]);

  let activeCameras = 0;
  if (cams.status === "fulfilled") {
    activeCameras = (cams.value as any[]).filter((c) => c.running).length;
  }

  let liveAlerts = 0;
  if (logsRes.status === "fulfilled") {
    const now = Date.now();
    liveAlerts = (logsRes.value as any[]).filter((l) => {
      const t = new Date(l.triggered_at).getTime();
      return now - t <= 15 * 60 * 1000;
    }).length;
  }

  let anomalies = 0;
  try {
    const anomalyResult = await api.getAnomalies();
    anomalies = anomalyResult.count || 0;
  } catch {
    anomalies = 0;
  }

  let detectionsToday = 0;
  if (detsRes.status === "fulfilled") {
    detectionsToday = (detsRes.value as any[]).length;
  }

  return { activeCameras, liveAlerts, anomalies, detectionsToday };
};

const CARD_CONFIGS = [
  {
    key: "activeCameras",
    title: "Active Cameras",
    icon: Camera,
    isPrimary: true,
    subtitle: "Currently streaming",
    accent: "#16A34A",
    iconBg: "rgba(22,163,74,0.15)",
  },
  {
    key: "liveAlerts",
    title: "Live Alerts",
    icon: AlertTriangle,
    subtitle: "Last 15 minutes",
    accent: "#D97706",
    iconBg: "rgba(217,119,6,0.10)",
  },
  {
    key: "anomalies",
    title: "Anomalies",
    icon: TrendingUp,
    subtitle: "Detected today",
    accent: "#DC2626",
    iconBg: "rgba(220,38,38,0.10)",
  },
  {
    key: "detectionsToday",
    title: "Detections",
    icon: Activity,
    subtitle: "Today's count",
    accent: "#0891B2",
    iconBg: "rgba(8,145,178,0.10)",
  },
];

export function SummaryCards() {
  const { data, isLoading } = useSWR("summaryStats", fetchSummaryStats, {
    refreshInterval: 10000,
  });

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: "var(--gap-island)",
      }}
      className="summary-cards-grid"
    >
      {CARD_CONFIGS.map((cfg) => {
        const Icon = cfg.icon;
        const value = data?.[cfg.key as keyof typeof data] ?? 0;

        return (
          <div
            key={cfg.key}
            style={{
              background: cfg.isPrimary
                ? "linear-gradient(135deg, #15803D 0%, #16A34A 100%)"
                : "var(--color-surface)",
              border: `1px solid ${cfg.isPrimary ? "#15803D" : "var(--color-border)"}`,
              borderRadius: "var(--radius-lg)",
              padding: "20px 20px",
              boxShadow: "var(--shadow-sm)",
              display: "flex",
              flexDirection: "column",
              gap: "12px",
              position: "relative",
              overflow: "hidden",
              transition: "box-shadow 150ms ease, transform 150ms ease",
            }}
            className="stat-card"
          >
            {/* Top-left border accent (non-primary) */}
            {!cfg.isPrimary && (
              <div style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: "3px",
                height: "100%",
                background: cfg.accent,
                borderRadius: "var(--radius-lg) 0 0 var(--radius-lg)",
                opacity: 0.7,
              }} />
            )}

            {/* Row: label + icon */}
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
              <p style={{
                fontSize: "0.8rem",
                fontWeight: 500,
                color: cfg.isPrimary ? "rgba(255,255,255,0.8)" : "var(--color-text-muted)",
                letterSpacing: "0.01em",
              }}>
                {cfg.title}
              </p>
              <div style={{
                width: "32px",
                height: "32px",
                borderRadius: "var(--radius-md)",
                background: cfg.isPrimary ? "rgba(255,255,255,0.15)" : cfg.iconBg,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}>
                <Icon style={{ width: "15px", height: "15px", color: cfg.isPrimary ? "white" : cfg.accent }} />
              </div>
            </div>

            {/* Value + subtitle */}
            <div>
              <p style={{
                fontSize: "2rem",
                fontWeight: 700,
                lineHeight: 1,
                color: cfg.isPrimary ? "white" : "var(--color-text)",
                fontFamily: "var(--font-ui)",
              }}>
                {isLoading ? "—" : value}
              </p>
              <p style={{
                fontSize: "0.6875rem",
                color: cfg.isPrimary ? "rgba(255,255,255,0.65)" : "var(--color-text-faint)",
                marginTop: "4px",
              }}>
                {cfg.subtitle}
              </p>
            </div>
          </div>
        );
      })}

      <style>{`
        @media (max-width: 900px) {
          .summary-cards-grid { grid-template-columns: repeat(2, 1fr) !important; }
        }
        @media (max-width: 560px) {
          .summary-cards-grid { grid-template-columns: 1fr !important; }
        }
        .stat-card:hover {
          box-shadow: var(--shadow-md) !important;
          transform: translateY(-1px);
        }
      `}</style>
    </div>
  );
}
