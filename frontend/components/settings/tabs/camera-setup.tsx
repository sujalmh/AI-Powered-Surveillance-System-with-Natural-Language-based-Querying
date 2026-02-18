"use client";

import { useEffect, useMemo, useState } from "react";
import { Plus, Trash2, Play, Square, Wifi, WifiOff } from "lucide-react";
import { api } from "@/lib/api";

type CameraDoc = {
  camera_id: number;
  source?: string;
  location?: string;
  status?: string; // "active" | "inactive" | "error"
  last_seen?: string;
  running?: boolean; // injected by backend runtime
  last_error?: string;
};

export function CameraSetup() {
  const [cameras, setCameras] = useState<CameraDoc[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  // form state
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
    setLoading(true);
    setErr(null);
    try {
      const list = await api.listCameras();
      const cams: CameraDoc[] = (list as any[]).map((c) => ({
        camera_id: Number(c.camera_id),
        source: c.source,
        location: c.location,
        status: c.status,
        last_seen: c.last_seen,
        running: Boolean(c.running),
        last_error: c.last_error,
      }));
      setCameras(cams);
    } catch (e: any) {
      setErr(e?.message || "Failed to load cameras");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const canAdd = useMemo(() => {
    return cameraId !== "" && String(source).trim().length > 0;
  }, [cameraId, source]);

  const probeSource = async (src: string | number) => {
    try {
      const res = await api.probeCamera(src, 3);
      return res;
    } catch (e: any) {
      return { ok: false, message: e?.message || "Probe request failed" };
    }
  };

  const onAddCamera = async () => {
    if (!canAdd) return;
    setSaving(true);
    setErr(null);
    setInfo(null);
    setProbingId("new");
    try {
      // 1) register camera in DB
      await api.registerCamera({
        camera_id: Number(cameraId),
        source: source.trim(),
        location: location.trim() || undefined,
      });

      // 2) probe source quickly
      const probe = await probeSource(source.trim());
      if (!probe.ok) {
        setErr(`Probe failed: ${probe.message}`);
        setInfo(null);
        // refresh list; backend may store last_error during /start, but probe doesn't. We still fetch fresh list.
        await load();
        return;
      }
      setInfo("Probe OK: camera is reachable. Starting stream...");

      // 3) start detection; backend checks by default (check=True)
      await api.startCamera(Number(cameraId), {
        source: source.trim(),
        location: location.trim() || undefined,
        show_window: false,
      });

      // 4) refresh list and reset form
      await load();
      setCameraId("");
      setSource("");
      setLocation("");
      setInfo("Camera started successfully.");
    } catch (e: any) {
      setErr(e?.message || "Failed to add/start camera");
    } finally {
      setProbingId(null);
      setSaving(false);
    }
  };

  const onStart = async (id: number, src?: string, loc?: string) => {
    setStartingId(id);
    setErr(null);
    setInfo(null);
    try {
      // probe first for immediate feedback
      setProbingId(id);
      const probe = await probeSource(src ?? id);
      setProbingId(null);
      if (!probe.ok) {
        setErr(`Probe failed: ${probe.message}`);
        return;
      }
      await api.startCamera(id, {
        // If a camera has no stored source, pass current source; otherwise rely on DB value
        source: src,
        location: loc,
        show_window: false,
      });
      await load();
      setInfo(`Camera #${id} started.`);
    } catch (e: any) {
      setErr(e?.message || "Failed to start camera");
    } finally {
      setStartingId(null);
    }
  };

  const onStop = async (id: number) => {
    setStoppingId(id);
    setErr(null);
    setInfo(null);
    try {
      await api.stopCamera(id);
      await load();
      setInfo(`Camera #${id} stopped.`);
    } catch (e: any) {
      setErr(e?.message || "Failed to stop camera");
    } finally {
      setStoppingId(null);
    }
  };

  const onProbeExisting = async (id: number, src?: string) => {
    setProbingId(id);
    setErr(null);
    setInfo(null);
    try {
      const probe = await probeSource(src ?? id);
      if (probe.ok) {
        setInfo(`Probe OK for camera #${id}: ${probe.message}`);
      } else {
        setErr(`Probe failed for camera #${id}: ${probe.message}`);
      }
    } catch (e: any) {
      setErr(e?.message || "Probe failed");
    } finally {
      setProbingId(null);
    }
  };

  const onDeleteClick = (id: number) => {
    setCameraToDelete(id);
    setShowDeleteModal(true);
  };

  const onDeleteConfirm = async () => {
    if (cameraToDelete === null) return;
    setDeletingId(cameraToDelete);
    setErr(null);
    setInfo(null);
    setShowDeleteModal(false);
    try {
      await api.deleteCamera(cameraToDelete);
      await load();
      setInfo(`Camera #${cameraToDelete} deleted successfully.`);
    } catch (e: any) {
      setErr(e?.message || "Failed to delete camera");
    } finally {
      setDeletingId(null);
      setCameraToDelete(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">Add / Register Camera</h3>
          {loading && <span className="text-xs text-muted-foreground">Loading…</span>}
        </div>
        {err && <div className="mb-3 text-xs text-red-400">{err}</div>}
        {info && <div className="mb-3 text-xs text-cyan-300">{info}</div>}
        <div className="grid grid-cols-3 gap-4">
          <input
            type="number"
            min={0}
            placeholder="Camera ID (e.g., 1)"
            value={cameraId}
            onChange={(e) => setCameraId(e.target.value === "" ? "" : Number(e.target.value))}
            className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none"
          />
          <input
            type="text"
            placeholder="Source (RTSP/HTTP URL or index e.g., 0)"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none"
          />
          <input
            type="text"
            placeholder="Location (optional)"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none"
          />
        </div>
        <div className="flex gap-2 mt-4">
          <button
            onClick={onAddCamera}
            disabled={!canAdd || saving || probingId === "new"}
            className="glow-button flex-1 flex items-center justify-center gap-2 px-4 py-2.5"
            title="Register, probe, then start camera"
          >
            <Plus className="w-4 h-4" />
            {probingId === "new" ? "Probing..." : saving ? "Registering..." : "Register & Start"}
          </button>
          <button
            onClick={async () => {
              if (!canAdd) return;
              setErr(null);
              setInfo(null);
              setProbingId("new");
              const res = await probeSource(source.trim());
              setProbingId(null);
              if (res.ok) setInfo(`Probe OK: ${res.message}`);
              else setErr(`Probe failed: ${res.message}`);
            }}
            disabled={!canAdd || probingId === "new" || saving}
            className="px-3 py-2 rounded-lg bg-white/10 border border-white/20 text-xs text-foreground hover:bg-white/15 disabled:opacity-60 flex items-center gap-2"
            title="Only probe without starting"
          >
            {probingId === "new" ? <WifiOff className="w-4 h-4" /> : <Wifi className="w-4 h-4" />}
            Probe Only
          </button>
        </div>
      </div>

      <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">Registered Cameras</h3>
          <button
            onClick={load}
            className="px-3 py-1.5 rounded-lg bg-white/10 border border-white/20 text-xs text-foreground hover:bg-white/15"
          >
            Refresh
          </button>
        </div>
        <div className="space-y-3">
          {cameras.map((camera) => {
            const isRunning = !!camera.running;
            const isError = camera.status === "error" || !!camera.last_error;
            return (
              <div
                key={camera.camera_id}
                className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-4"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium text-foreground">
                      Camera #{camera.camera_id} {camera.location ? <span className="text-muted-foreground">· {camera.location}</span> : null}
                    </p>
                    <p className="text-xs text-muted-foreground mt-1">{camera.source || "-"}</p>
                    <p className="text-xs mt-1 flex items-center gap-2">
                      <span
                        className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                          isRunning
                            ? "bg-green-500/20 text-green-400 border border-green-500/50"
                            : isError
                            ? "bg-red-500/20 text-red-400 border border-red-500/50"
                            : "bg-gray-500/20 text-gray-400 border border-gray-500/50"
                        }`}
                      >
                        {isRunning ? "Running" : isError ? "Error" : "Stopped"}
                      </span>
                      {camera.last_seen && (
                        <span className="text-muted-foreground">Last seen: {new Date(camera.last_seen).toLocaleString()}</span>
                      )}
                    </p>
                    {camera.last_error && (
                      <p className="text-xs text-red-400 mt-1 break-words">Last error: {camera.last_error}</p>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => onProbeExisting(camera.camera_id, camera.source)}
                      disabled={probingId === camera.camera_id}
                      className="p-2 hover:bg-white/10 rounded transition-colors disabled:opacity-60 flex items-center gap-1"
                      title="Probe connectivity"
                    >
                      {probingId === camera.camera_id ? <WifiOff className="w-4 h-4 text-yellow-400" /> : <Wifi className="w-4 h-4 text-cyan-400" />}
                      {probingId === camera.camera_id ? "Probing..." : "Probe"}
                    </button>
                    {isRunning ? (
                      <button
                        onClick={() => onStop(camera.camera_id)}
                        disabled={stoppingId === camera.camera_id}
                        className="px-3 py-2 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 hover:bg-red-500/20 hover:border-red-500/50 transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer flex items-center gap-1.5 font-medium text-sm"
                        title="Stop"
                      >
                        <Square className="w-4 h-4" />
                        {stoppingId === camera.camera_id ? "Stopping..." : "Stop"}
                      </button>
                    ) : (
                      <button
                        onClick={() => onStart(camera.camera_id, camera.source, camera.location)}
                        disabled={startingId === camera.camera_id}
                        className="px-3 py-2 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 hover:bg-green-500/20 hover:border-green-500/50 transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer flex items-center gap-1.5 font-medium text-sm"
                        title="Start"
                      >
                        <Play className="w-4 h-4" />
                        {startingId === camera.camera_id ? "Starting..." : "Start"}
                      </button>
                    )}
                    <button
                      onClick={() => onDeleteClick(camera.camera_id)}
                      disabled={deletingId === camera.camera_id}
                      className="p-2 hover:bg-red-500/10 rounded transition-colors disabled:opacity-60 flex items-center gap-1"
                      title="Delete camera"
                    >
                      <Trash2 className="w-4 h-4 text-red-400" />
                      {deletingId === camera.camera_id ? "Deleting..." : ""}
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
          {cameras.length === 0 && !loading && (
            <div className="text-sm text-muted-foreground">No cameras registered yet.</div>
          )}
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {showDeleteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 dark:bg-black/60 backdrop-blur-sm">
          <div className="bg-gradient-to-br from-white to-gray-50 dark:from-gray-900/95 dark:to-black/95 backdrop-blur-xl border border-gray-300 dark:border-white/20 rounded-2xl p-6 max-w-md w-full mx-4 shadow-2xl">
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Delete Camera</h3>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              Are you sure you want to delete Camera #{cameraToDelete}? This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => {
                  setShowDeleteModal(false);
                  setCameraToDelete(null);
                }}
                className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-white/10 border border-gray-300 dark:border-white/20 text-gray-700 dark:text-white hover:bg-gray-300 dark:hover:bg-white/15 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={onDeleteConfirm}
                className="px-4 py-2 rounded-lg bg-red-100 dark:bg-red-500/20 border border-red-300 dark:border-red-500/50 text-red-600 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-500/30 transition-colors"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
