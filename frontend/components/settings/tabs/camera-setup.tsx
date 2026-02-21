"use client";

import { useEffect, useMemo, useState } from "react";
import { Plus, Trash2, Play, Square, Wifi, WifiOff } from "lucide-react";
import { api } from "@/lib/api";

type CameraDoc = {
  camera_id: number;
  source?: string;
  location?: string;
  status?: string;
  last_seen?: string;
  running?: boolean;
  last_error?: string;
};

const inputStyle: React.CSSProperties = {
  width: "100%", padding: "8px 11px",
  border: "1px solid var(--color-border)", borderRadius: "var(--radius-md)",
  background: "var(--color-surface-raised)", color: "var(--color-text)",
  fontSize: "0.875rem", fontFamily: "var(--font-ui)",
  outline: "none", transition: "border-color 150ms",
};

export function CameraSetup() {
  const [cameras, setCameras] = useState<CameraDoc[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [cameraId, setCameraId] = useState<number | "">("");
  const [source, setSource] = useState<string>("");
  const [location, setLocation] = useState<string>("");
  const [saving, setSaving] = useState(false);
  const [startingId, setStartingId] = useState<number | null>(null);
  const [stoppingId, setStoppingId] = useState<number | null>(null);
  const [probingId, setProbingId] = useState<number | "new" | null>(null);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [cameraToDelete, setCameraToDelete] = useState<number | null>(null);

  const load = async () => {
    setLoading(true); setErr(null);
    try {
      const list = await api.listCameras();
      setCameras((list as any[]).map((c) => ({
        camera_id: Number(c.camera_id), source: c.source, location: c.location,
        status: c.status, last_seen: c.last_seen, running: Boolean(c.running), last_error: c.last_error,
      })));
    } catch (e: any) { setErr(e?.message || "Failed to load cameras"); }
    finally { setLoading(false); }
  };

  useEffect(() => { load(); }, []);
  const canAdd = useMemo(() => cameraId !== "" && String(source).trim().length > 0, [cameraId, source]);

  const probeSource = async (src: string | number) => {
    try { return await api.probeCamera(src, 3); }
    catch (e: any) { return { ok: false, message: e?.message || "Probe request failed" }; }
  };

  const onAddCamera = async () => {
    if (!canAdd) return;
    setSaving(true); setErr(null); setInfo(null); setProbingId("new");
    try {
      await api.registerCamera({ camera_id: Number(cameraId), source: source.trim(), location: location.trim() || undefined });
      const probe = await probeSource(source.trim());
      if (!probe.ok) { setErr(`Probe failed: ${probe.message}`); await load(); return; }
      setInfo("Probe OK: camera is reachable. Starting stream...");
      await api.startCamera(Number(cameraId), { source: source.trim(), location: location.trim() || undefined, show_window: false });
      await load(); setCameraId(""); setSource(""); setLocation(""); setInfo("Camera started successfully.");
    } catch (e: any) { setErr(e?.message || "Failed to add/start camera"); }
    finally { setProbingId(null); setSaving(false); }
  };

  const onStart = async (id: number, src?: string, loc?: string) => {
    setStartingId(id); setErr(null); setInfo(null);
    try {
      setProbingId(id); const probe = await probeSource(src ?? id); setProbingId(null);
      if (!probe.ok) { setErr(`Probe failed: ${probe.message}`); return; }
      await api.startCamera(id, { source: src, location: loc, show_window: false });
      await load(); setInfo(`Camera #${id} started.`);
    } catch (e: any) { setErr(e?.message || "Failed to start camera"); }
    finally { setStartingId(null); }
  };

  const onStop = async (id: number) => {
    setStoppingId(id); setErr(null); setInfo(null);
    try { await api.stopCamera(id); await load(); setInfo(`Camera #${id} stopped.`); }
    catch (e: any) { setErr(e?.message || "Failed to stop camera"); }
    finally { setStoppingId(null); }
  };

  const onProbeExisting = async (id: number, src?: string) => {
    setProbingId(id); setErr(null); setInfo(null);
    try {
      const probe = await probeSource(src ?? id);
      if (probe.ok) setInfo(`Probe OK for camera #${id}: ${probe.message}`);
      else setErr(`Probe failed for camera #${id}: ${probe.message}`);
    } catch (e: any) { setErr(e?.message || "Probe failed"); }
    finally { setProbingId(null); }
  };

  const onDeleteConfirm = async () => {
    if (cameraToDelete === null) return;
    setDeletingId(cameraToDelete); setErr(null); setInfo(null); setShowDeleteModal(false);
    try { await api.deleteCamera(cameraToDelete); await load(); setInfo(`Camera #${cameraToDelete} deleted successfully.`); }
    catch (e: any) { setErr(e?.message || "Failed to delete camera"); }
    finally { setDeletingId(null); setCameraToDelete(null); }
  };

  const focusBorder = (e: React.FocusEvent<HTMLInputElement>) => (e.target.style.borderColor = "var(--color-primary)");
  const blurBorder = (e: React.FocusEvent<HTMLInputElement>) => (e.target.style.borderColor = "var(--color-border)");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      {/* Add Camera island */}
      <div style={{
        background: "var(--color-surface)", border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)", padding: "20px 24px", boxShadow: "var(--shadow-sm)",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
          <h3 style={{ fontSize: "0.9375rem", fontWeight: 700, color: "var(--color-text)" }}>Add / Register Camera</h3>
          {loading && <span style={{ fontSize: "0.75rem", color: "var(--color-text-faint)" }}>Loading…</span>}
        </div>

        {err && <div style={{ marginBottom: 10, padding: "8px 12px", background: "rgba(220,38,38,0.08)", border: "1px solid rgba(220,38,38,0.2)", borderRadius: "var(--radius-md)", fontSize: "0.8125rem", color: "var(--color-danger)" }}>{err}</div>}
        {info && <div style={{ marginBottom: 10, padding: "8px 12px", background: "var(--color-primary-light)", border: "1px solid rgba(22,163,74,0.2)", borderRadius: "var(--radius-md)", fontSize: "0.8125rem", color: "var(--color-primary)" }}>{info}</div>}

        <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr 1.5fr", gap: "10px" }}>
          <input type="number" min={0} placeholder="Camera ID (e.g., 1)" value={cameraId}
            onChange={(e) => setCameraId(e.target.value === "" ? "" : Number(e.target.value))}
            style={inputStyle} onFocus={focusBorder} onBlur={blurBorder} />
          <input type="text" placeholder="Source (RTSP/HTTP URL or index)" value={source}
            onChange={(e) => setSource(e.target.value)} style={inputStyle} onFocus={focusBorder} onBlur={blurBorder} />
          <input type="text" placeholder="Location (optional)" value={location}
            onChange={(e) => setLocation(e.target.value)} style={inputStyle} onFocus={focusBorder} onBlur={blurBorder} />
        </div>

        <div style={{ display: "flex", gap: "8px", marginTop: "12px" }}>
          <button
            onClick={onAddCamera}
            disabled={!canAdd || saving || probingId === "new"}
            style={{
              flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: "6px",
              padding: "9px 16px", borderRadius: "var(--radius-md)", border: "none",
              background: "var(--color-primary)", color: "#fff",
              fontSize: "0.875rem", fontWeight: 600, cursor: "pointer",
              opacity: (!canAdd || saving || probingId === "new") ? 0.5 : 1, transition: "opacity 150ms",
            }}
          >
            <Plus style={{ width: 14, height: 14 }} />
            {probingId === "new" ? "Probing..." : saving ? "Registering..." : "Register & Start"}
          </button>
          <button
            onClick={async () => {
              if (!canAdd) return;
              setErr(null); setInfo(null); setProbingId("new");
              const res = await probeSource(source.trim()); setProbingId(null);
              if (res.ok) setInfo(`Probe OK: ${res.message}`);
              else setErr(`Probe failed: ${res.message}`);
            }}
            disabled={!canAdd || probingId === "new" || saving}
            style={{
              display: "flex", alignItems: "center", gap: "6px",
              padding: "9px 14px", borderRadius: "var(--radius-md)",
              border: "1px solid var(--color-border)", background: "var(--color-surface-raised)",
              color: "var(--color-text-muted)", fontSize: "0.8125rem", fontWeight: 500,
              cursor: "pointer", opacity: (!canAdd || probingId === "new" || saving) ? 0.5 : 1,
              transition: "all 150ms",
            }}
          >
            {probingId === "new" ? <WifiOff style={{ width: 14, height: 14 }} /> : <Wifi style={{ width: 14, height: 14 }} />}
            Probe Only
          </button>
        </div>
      </div>

      {/* Registered cameras island */}
      <div style={{
        background: "var(--color-surface)", border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)", padding: "20px 24px", boxShadow: "var(--shadow-sm)",
      }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "16px" }}>
          <h3 style={{ fontSize: "0.9375rem", fontWeight: 700, color: "var(--color-text)" }}>Registered Cameras</h3>
          <button
            onClick={load}
            style={{
              padding: "6px 12px", borderRadius: "var(--radius-md)",
              border: "1px solid var(--color-border)", background: "var(--color-surface-raised)",
              color: "var(--color-text-muted)", fontSize: "0.8125rem", cursor: "pointer", transition: "all 150ms",
            }}
          >Refresh</button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {cameras.map((camera) => {
            const isRunning = !!camera.running;
            const isError = camera.status === "error" || !!camera.last_error;
            const statusColor = isRunning ? "#16A34A" : isError ? "var(--color-danger)" : "var(--color-text-faint)";
            const statusBg = isRunning ? "rgba(22,163,74,0.10)" : isError ? "rgba(220,38,38,0.10)" : "var(--color-surface-raised)";

            return (
              <div key={camera.camera_id} style={{
                padding: "12px 14px", borderRadius: "var(--radius-md)",
                border: "1px solid var(--color-border)", background: "var(--color-surface-raised)",
                display: "flex", alignItems: "center", justifyContent: "space-between", gap: "12px",
              }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-text)" }}>
                    Camera #{camera.camera_id}
                    {camera.location && <span style={{ color: "var(--color-text-muted)", fontWeight: 400 }}> · {camera.location}</span>}
                  </p>
                  <p style={{ fontSize: "0.75rem", color: "var(--color-text-faint)", marginTop: 2 }}>{camera.source || "–"}</p>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", marginTop: 4 }}>
                    <span style={{
                      padding: "2px 8px", borderRadius: "var(--radius-full)",
                      fontSize: "0.6875rem", fontWeight: 700,
                      color: statusColor, background: statusBg,
                    }}>
                      {isRunning ? "Running" : isError ? "Error" : "Stopped"}
                    </span>
                    {camera.last_seen && (
                      <span style={{ fontSize: "0.6875rem", color: "var(--color-text-faint)" }}>
                        Last seen: {new Date(camera.last_seen).toLocaleString()}
                      </span>
                    )}
                  </div>
                  {camera.last_error && (
                    <p style={{ fontSize: "0.75rem", color: "var(--color-danger)", marginTop: 3, wordBreak: "break-all" }}>
                      Error: {camera.last_error}
                    </p>
                  )}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", flexShrink: 0 }}>
                  <button
                    onClick={() => onProbeExisting(camera.camera_id, camera.source)}
                    disabled={probingId === camera.camera_id}
                    title="Probe connectivity"
                    style={{
                      display: "flex", alignItems: "center", gap: "4px",
                      padding: "6px 10px", borderRadius: "var(--radius-sm)",
                      border: "1px solid var(--color-border)", background: "transparent",
                      color: "var(--color-text-muted)", fontSize: "0.75rem", cursor: "pointer",
                      opacity: probingId === camera.camera_id ? 0.6 : 1, transition: "all 150ms",
                    }}
                  >
                    {probingId === camera.camera_id ? <WifiOff style={{ width: 13, height: 13 }} /> : <Wifi style={{ width: 13, height: 13 }} />}
                    {probingId === camera.camera_id ? "Probing…" : "Probe"}
                  </button>

                  {isRunning ? (
                    <button
                      onClick={() => onStop(camera.camera_id)}
                      disabled={stoppingId === camera.camera_id}
                      style={{
                        display: "flex", alignItems: "center", gap: "4px",
                        padding: "6px 12px", borderRadius: "var(--radius-sm)",
                        border: "1px solid rgba(220,38,38,0.25)", background: "rgba(220,38,38,0.08)",
                        color: "var(--color-danger)", fontSize: "0.75rem", fontWeight: 600,
                        cursor: "pointer", opacity: stoppingId === camera.camera_id ? 0.6 : 1, transition: "all 150ms",
                      }}
                    >
                      <Square style={{ width: 12, height: 12 }} />
                      {stoppingId === camera.camera_id ? "Stopping…" : "Stop"}
                    </button>
                  ) : (
                    <button
                      onClick={() => onStart(camera.camera_id, camera.source, camera.location)}
                      disabled={startingId === camera.camera_id}
                      style={{
                        display: "flex", alignItems: "center", gap: "4px",
                        padding: "6px 12px", borderRadius: "var(--radius-sm)",
                        border: "1px solid rgba(22,163,74,0.25)", background: "rgba(22,163,74,0.08)",
                        color: "var(--color-primary)", fontSize: "0.75rem", fontWeight: 600,
                        cursor: "pointer", opacity: startingId === camera.camera_id ? 0.6 : 1, transition: "all 150ms",
                      }}
                    >
                      <Play style={{ width: 12, height: 12 }} />
                      {startingId === camera.camera_id ? "Starting…" : "Start"}
                    </button>
                  )}

                  <button
                    onClick={() => { setCameraToDelete(camera.camera_id); setShowDeleteModal(true); }}
                    disabled={deletingId === camera.camera_id}
                    title="Delete camera"
                    style={{
                      width: 30, height: 30, display: "flex", alignItems: "center", justifyContent: "center",
                      borderRadius: "var(--radius-sm)", border: "none", background: "transparent",
                      color: "var(--color-text-faint)", cursor: "pointer",
                      opacity: deletingId === camera.camera_id ? 0.6 : 1, transition: "all 150ms",
                    }}
                    onMouseEnter={e => { e.currentTarget.style.color = "var(--color-danger)"; e.currentTarget.style.background = "rgba(220,38,38,0.08)"; }}
                    onMouseLeave={e => { e.currentTarget.style.color = "var(--color-text-faint)"; e.currentTarget.style.background = "transparent"; }}
                  >
                    <Trash2 style={{ width: 13, height: 13 }} />
                  </button>
                </div>
              </div>
            );
          })}
          {cameras.length === 0 && !loading && (
            <div style={{ padding: "32px 16px", textAlign: "center", color: "var(--color-text-faint)", fontSize: "0.875rem" }}>
              No cameras registered yet.
            </div>
          )}
        </div>
      </div>

      {/* Delete modal */}
      {showDeleteModal && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 50,
          display: "flex", alignItems: "center", justifyContent: "center",
          background: "rgba(0,0,0,0.45)", backdropFilter: "blur(2px)",
        }}>
          <div style={{
            background: "var(--color-surface)", border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-lg)", padding: "24px 28px",
            maxWidth: 420, width: "90%", boxShadow: "var(--shadow-lg)",
          }}>
            <h3 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--color-text)", marginBottom: 8 }}>Delete Camera</h3>
            <p style={{ fontSize: "0.875rem", color: "var(--color-text-muted)", marginBottom: "24px" }}>
              Are you sure you want to delete Camera #{cameraToDelete}? This action cannot be undone.
            </p>
            <div style={{ display: "flex", gap: "8px", justifyContent: "flex-end" }}>
              <button
                onClick={() => { setShowDeleteModal(false); setCameraToDelete(null); }}
                style={{
                  padding: "8px 16px", borderRadius: "var(--radius-md)",
                  border: "1px solid var(--color-border)", background: "var(--color-surface-raised)",
                  color: "var(--color-text)", fontSize: "0.875rem", fontWeight: 500, cursor: "pointer",
                }}
              >Cancel</button>
              <button
                onClick={onDeleteConfirm}
                style={{
                  padding: "8px 16px", borderRadius: "var(--radius-md)",
                  border: "1px solid rgba(220,38,38,0.3)", background: "rgba(220,38,38,0.1)",
                  color: "var(--color-danger)", fontSize: "0.875rem", fontWeight: 600, cursor: "pointer",
                }}
              >Delete</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
