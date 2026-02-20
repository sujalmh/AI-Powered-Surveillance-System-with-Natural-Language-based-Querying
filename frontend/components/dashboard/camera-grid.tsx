"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { api, API_BASE } from "@/lib/api";

type CameraDoc = {
  camera_id: number;
  source?: string;
  location?: string;
  status?: string;
  last_seen?: string;
  running?: boolean;
};

type ZoneInfo = {
  zone_id: string;
  name: string;
  count: number;
  capacity?: number;
  occupancy_pct?: number;
};

type Card = {
  id: number;
  name: string;
  status: "live" | "recording" | "stopped";
  people: number;
  zones?: ZoneInfo[];
};

/* ────────────────────────────────── CameraTile ────────────────────────────────── */

type CameraTileProps = { id: number; name: string; status: "live" | "recording" | "stopped"; people: number; zones?: ZoneInfo[] };

function CameraTile({ id, name, status, people, zones }: CameraTileProps) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [streamOk, setStreamOk] = useState(true);
  const retryTimer = useRef<NodeJS.Timeout | null>(null);

  // The MJPEG multipart stream URL — browsers render this natively in an <img> tag.
  // Each frame is pushed by the server; no polling needed.
  const streamUrl = `${API_BASE}/api/videos/stream/${id}`;

  // On error: wait 3 seconds then retry the stream (don't fall back to polling)
  const handleError = () => {
    setStreamOk(false);
    if (retryTimer.current) clearTimeout(retryTimer.current);
    retryTimer.current = setTimeout(() => {
      setStreamOk(true);
    }, 3000);
  };

  // When streamOk flips back to true, force the img src to reload the stream
  useEffect(() => {
    if (streamOk && imgRef.current) {
      // Adding a cache-buster on retry to force a fresh connection
      imgRef.current.src = `${streamUrl}?t=${Date.now()}`;
    }
  }, [streamOk, streamUrl]);

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (retryTimer.current) clearTimeout(retryTimer.current);
    };
  }, []);

  return (
    <div className="group relative overflow-hidden rounded-lg bg-black/50 aspect-video">
      {streamOk ? (
        <img
          ref={imgRef}
          src={streamUrl}
          alt={`Camera ${name}`}
          className="absolute inset-0 w-full h-full object-cover"
          onError={handleError}
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="flex flex-col items-center gap-2">
            <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            <span className="text-xs text-white/60">Reconnecting…</span>
          </div>
        </div>
      )}
      <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
      <div className="absolute bottom-0 left-0 right-0 p-3 translate-y-full group-hover:translate-y-0 transition-transform">
        <p className="text-sm font-semibold text-white">{name}</p>
        <div className="flex items-center justify-between mt-2">
          <span
            className={`text-xs px-2 py-1 rounded ${
              status === "live"
                ? "bg-red-500/80 text-white"
                : status === "recording"
                ? "bg-blue-500/80 text-white"
                : "bg-gray-500/80 text-white"
            }`}
          >
            {status === "live" ? "LIVE" : status === "recording" ? "REC" : "STOPPED"}
          </span>
          <span className="text-xs text-white/90">{people} people</span>
        </div>
        {zones && zones.length > 0 && (
          <div className="mt-2 space-y-1">
            {zones.map((z) => (
              <div key={z.zone_id} className="flex items-center justify-between text-xs text-white/90">
                <span>{z.name}: {z.count}{z.capacity != null ? ` / ${z.capacity}` : ""}</span>
                {z.occupancy_pct != null && (
                  <div className="w-12 h-1.5 bg-white/20 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-white/90 rounded-full"
                      style={{ width: `${Math.min(100, z.occupancy_pct)}%` }}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ────────────────────────────────── CameraGrid ────────────────────────────────── */

export function CameraGrid() {
  const [cards, setCards] = useState<Card[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setErr(null);
    try {
      const cams = (await api.listCameras()) as any[];
      // Build base cards from camera list
      const base: Card[] = cams.map((c) => ({
        id: Number(c.camera_id),
        name: c.location ? `#${c.camera_id} · ${c.location}` : `#${c.camera_id}`,
        status: c.running ? "live" : "stopped",
        people: 0,
      }));

      // Enrich with occupancy (people count + zone counts/capacity) or fallback to detections
      for (let i = 0; i < base.length; i++) {
        try {
          const occ = await api.getOccupancy(base[i].id) as any;
          base[i].people = occ?.person_count ?? 0;
          if (Array.isArray(occ?.zones) && occ.zones.length > 0) {
            base[i].zones = occ.zones.map((z: any) => ({
              zone_id: z.zone_id,
              name: z.name,
              count: z.count ?? 0,
              capacity: z.capacity,
              occupancy_pct: z.occupancy_pct,
            }));
          }
        } catch {
          try {
            const p = new URLSearchParams();
            p.set("camera_id", String(base[i].id));
            p.set("object", "person");
            p.set("last_minutes", "5");
            p.set("limit", "50");
            const dets = (await api.listDetections(p)) as any[];
            const count = dets.reduce((acc: number, d: any) => {
              const objs = Array.isArray(d.objects) ? d.objects : [];
              return acc + objs.filter((o: any) => o.object_name === "person").length;
            }, 0);
            base[i].people = count;
          } catch {
            // ignore per-camera error
          }
        }
      }

      setCards(base);
    } catch (e: any) {
      setErr(e?.message || "Failed to load cameras");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-foreground">Live Camera Feed</h2>
        {loading && <span className="text-xs text-muted-foreground">Loading…</span>}
        {err && <span className="text-xs text-red-400">{err}</span>}
        <button
          onClick={load}
          className="ml-auto px-3 py-1.5 rounded-lg bg-white/10 border border-white/20 text-xs text-foreground hover:bg-white/15"
        >
          Refresh
        </button>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {cards.map((camera) => (
          <CameraTile
            key={camera.id}
            id={camera.id}
            name={camera.name}
            status={camera.status}
            people={camera.people}
            zones={camera.zones}
          />
        ))}
        {cards.length === 0 && !loading && (
          <div className="col-span-3 text-sm text-muted-foreground">No cameras registered yet.</div>
        )}
      </div>
    </div>
  );
}
