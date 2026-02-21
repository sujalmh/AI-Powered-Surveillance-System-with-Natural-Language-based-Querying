"use client";

import React, { useEffect, useRef, useState, memo } from "react";
import useSWR from "swr";
import { api, API_BASE } from "@/lib/api";
import { Users } from "lucide-react";

type ZoneInfo = {
  zone_id: string;
  name: string;
  count: number;
  capacity?: number;
};

type CameraCard = {
  id: number;
  name: string;
  location?: string;
  status: "live" | "stopped";
  people: number;
  zones?: ZoneInfo[];
};

const fetchCameras = async (): Promise<CameraCard[]> => {
  const cams = (await api.listCameras()) as any[];
  const base: CameraCard[] = cams.map((c) => ({
    id: Number(c.camera_id),
    name: c.location ? c.location : `Camera ${c.camera_id}`,
    location: c.location,
    status: c.running ? "live" : "stopped",
    people: 0,
  }));

  const fetchPromises = base.map(async (cam) => {
    try {
      const occ = (await api.getOccupancy(cam.id)) as any;
      cam.people = occ?.person_count ?? 0;
      if (Array.isArray(occ?.zones) && occ.zones.length > 0) {
        cam.zones = occ.zones.map((z: any) => ({
          zone_id: z.zone_id,
          name: z.name,
          count: z.count ?? 0,
          capacity: z.capacity,
        }));
      }
    } catch {
      try {
        const p = new URLSearchParams();
        p.set("camera_id", String(cam.id));
        p.set("object", "person");
        p.set("last_minutes", "5");
        p.set("limit", "50");
        const dets = (await api.listDetections(p)) as any[];
        cam.people = dets.reduce((acc: number, d: any) => {
          const objs = Array.isArray(d.objects) ? d.objects : [];
          return acc + objs.filter((o: any) => o.object_name === "person").length;
        }, 0);
      } catch {}
    }
    return cam;
  });

  return Promise.all(fetchPromises);
};

/* ────────────────────────────────────────────── */

const CameraTile = memo(({ id, name, status, people, zones }: CameraCard) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const [streamOk, setStreamOk] = useState(true);
  const retryTimer = useRef<NodeJS.Timeout | null>(null);
  const streamUrl = `${API_BASE}/api/videos/stream/${id}`;

  const handleError = () => {
    setStreamOk(false);
    if (retryTimer.current) clearTimeout(retryTimer.current);
    retryTimer.current = setTimeout(() => setStreamOk(true), 3000);
  };

  useEffect(() => {
    if (streamOk && imgRef.current) {
      imgRef.current.src = `${streamUrl}?t=${Date.now()}`;
    }
  }, [streamOk, streamUrl]);

  useEffect(() => () => { if (retryTimer.current) clearTimeout(retryTimer.current); }, []);

  const isLive = status === "live";

  return (
    <div
      style={{
        position: "relative",
        overflow: "hidden",
        borderRadius: "var(--radius-md)",
        background: "#0E100E",
        aspectRatio: "16/9",
        border: "1px solid rgba(255,255,255,0.07)",
        transition: "transform 150ms ease, box-shadow 150ms ease",
      }}
      className="camera-tile"
    >
      {/* Stream / fallback */}
      {streamOk ? (
        <img
          ref={imgRef}
          src={streamUrl}
          alt={name}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }}
          onError={handleError}
        />
      ) : (
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          gap: "8px", color: "#555",
        }}>
          <div style={{
            width: "24px", height: "24px",
            border: "3px solid #222",
            borderTopColor: "#444",
            borderRadius: "50%",
            animation: "spin 1s linear infinite",
          }} />
          <span style={{ fontSize: "0.6875rem", fontFamily: "var(--font-mono)", letterSpacing: "0.05em" }}>
            Reconnecting...
          </span>
        </div>
      )}

      {/* Top-right: status badge */}
      <div style={{ position: "absolute", top: "8px", right: "8px" }}>
        <span style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "4px",
          padding: "2px 8px",
          borderRadius: "var(--radius-full)",
          background: "rgba(0,0,0,0.55)",
          backdropFilter: "blur(4px)",
          WebkitBackdropFilter: "blur(4px)",
          fontSize: "0.625rem",
          fontFamily: "var(--font-mono)",
          fontWeight: 600,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: isLive ? "#4ADE80" : "#888",
          border: `1px solid ${isLive ? "rgba(74,222,128,0.3)" : "rgba(255,255,255,0.1)"}`,
        }}>
          {isLive && <span style={{ width: "5px", height: "5px", borderRadius: "50%", background: "#4ADE80", display: "inline-block", animation: "pulse 2s ease-in-out infinite" }} />}
          {isLive ? "LIVE" : "STOPPED"}
        </span>
      </div>

      {/* Bottom-left: camera label */}
      <div style={{ position: "absolute", bottom: "8px", left: "8px" }}>
        <span style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "5px",
          padding: "2px 8px",
          borderRadius: "var(--radius-full)",
          background: "rgba(0,0,0,0.65)",
          backdropFilter: "blur(4px)",
          WebkitBackdropFilter: "blur(4px)",
          fontSize: "0.6875rem",
          color: "#ddd",
          border: "1px solid rgba(255,255,255,0.1)",
        }}>
          <span style={{ fontFamily: "var(--font-mono)", color: "#888", fontSize: "0.625rem" }}>#{id}</span>
          <span>{name}</span>
        </span>
      </div>

      {/* Bottom-right: people chip */}
      <div style={{ position: "absolute", bottom: "8px", right: "8px" }}>
        <span style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "4px",
          padding: "2px 8px",
          borderRadius: "var(--radius-full)",
          background: "rgba(0,0,0,0.55)",
          backdropFilter: "blur(4px)",
          WebkitBackdropFilter: "blur(4px)",
          fontSize: "0.6875rem",
          color: "#ccc",
          border: "1px solid rgba(255,255,255,0.1)",
        }}>
          <Users style={{ width: "10px", height: "10px" }} />
          {people}
        </span>
      </div>

      {/* Zone overlay on hover */}
      {zones && zones.length > 0 && (
        <div style={{
          position: "absolute",
          bottom: "36px",
          left: "8px",
          background: "rgba(0,0,0,0.80)",
          backdropFilter: "blur(4px)",
          borderRadius: "var(--radius-sm)",
          padding: "6px 8px",
          display: "flex",
          flexDirection: "column",
          gap: "3px",
          opacity: 0,
          transition: "opacity 150ms ease",
        }} className="zone-overlay">
          {zones.map((z) => (
            <div key={z.zone_id} style={{ display: "flex", justifyContent: "space-between", gap: "12px" }}>
              <span style={{ fontSize: "0.625rem", color: "#aaa", fontFamily: "var(--font-mono)" }}>{z.name}</span>
              <span style={{ fontSize: "0.625rem", color: "#ddd", fontFamily: "var(--font-mono)" }}>
                {z.count}{z.capacity ? `/${z.capacity}` : ""}
              </span>
            </div>
          ))}
        </div>
      )}

      <style>{`
        .camera-tile:hover {
          transform: scale(1.01);
          box-shadow: 0 0 0 2px rgba(22,163,74,0.4);
        }
        .camera-tile:hover .zone-overlay {
          opacity: 1 !important;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
      `}</style>
    </div>
  );
});
CameraTile.displayName = "CameraTile";

/* ────────────────────────────────────────────── */

export function CameraGrid() {
  const { data: cards, error, isLoading } = useSWR("cameras", fetchCameras, { refreshInterval: 5000 });

  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-sm)",
        overflow: "hidden",
        flex: 1,
        display: "flex",
        flexDirection: "column",
      }}
    >
      {/* Header */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "14px 16px",
        borderBottom: "1px solid var(--color-border)",
      }}>
        <p style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--color-text)" }}>Live Camera Feed</p>
        <a href="#" style={{
          fontSize: "0.75rem",
          color: "var(--color-primary)",
          textDecoration: "none",
          display: "flex",
          alignItems: "center",
          gap: "3px",
        }}>
          View All →
        </a>
      </div>

      {/* Grid */}
      <div style={{ padding: "12px", flex: 1 }}>
        {error && (
          <p style={{ fontSize: "0.8125rem", color: "var(--color-danger)", marginBottom: "10px" }}>
            Error loading cameras: {error.message}
          </p>
        )}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: "10px",
        }} className="camera-grid-inner">
          {isLoading && !cards
            ? Array.from({ length: 6 }).map((_, i) => (
                <div key={i} style={{
                  background: "var(--color-surface-raised)",
                  borderRadius: "var(--radius-md)",
                  aspectRatio: "16/9",
                  animation: "pulse-bg 1.5s ease-in-out infinite",
                }} />
              ))
            : cards && cards.length > 0
            ? cards.map((cam) => <CameraTile key={cam.id} {...cam} />)
            : (
                <div style={{
                  gridColumn: "1/-1",
                  textAlign: "center",
                  padding: "40px 20px",
                  color: "var(--color-text-faint)",
                  fontSize: "0.875rem",
                }}>
                  No cameras registered yet.
                </div>
              )
          }
        </div>
      </div>

      <style>{`
        @media (max-width: 900px) {
          .camera-grid-inner { grid-template-columns: repeat(2, 1fr) !important; }
        }
        @media (max-width: 560px) {
          .camera-grid-inner { grid-template-columns: 1fr !important; }
        }
        @keyframes pulse-bg {
          0%,100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}
