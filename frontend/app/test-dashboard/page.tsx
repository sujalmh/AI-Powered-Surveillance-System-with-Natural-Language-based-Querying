"use client";

import React, { useState, useRef } from "react";
import { MainLayout } from "@/components/layout/main-layout";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Loader2, Trash2, Video, MessageSquare, FileVideo, CheckCircle2, AlertCircle, FlaskConical } from "lucide-react";
import { api, API_BASE } from "@/lib/api";
import { ChatInterface } from "@/components/conversation/chat-interface";

type IndexingResult = {
  ok: boolean;
  camera_id: number;
  clip_path: string;
  clip_url: string | null;
  indexing: Record<string, any>;
};

type VideoItem = {
  id: string;
  file: File;
  cameraId: number;
  status: 'pending' | 'uploading' | 'indexing' | 'completed' | 'error';
  result?: IndexingResult;
  error?: string;
  frames?: Array<Record<string, any>>;
};

const TABS = [
  { id: "videos", label: "Videos", icon: Video },
  { id: "results", label: "Indexing Results", icon: FileVideo },
  { id: "conversation", label: "Conversation", icon: MessageSquare },
];

const statusPill = (status: VideoItem["status"]) => {
  const map: Record<string, { label: string; color: string; bg: string; spin?: boolean }> = {
    pending:   { label: "Pending",   color: "var(--color-text-muted)",  bg: "var(--color-surface-raised)" },
    uploading: { label: "Uploading", color: "#16A34A",                  bg: "rgba(22,163,74,0.12)", spin: true },
    indexing:  { label: "Indexing",  color: "#7C3AED",                  bg: "rgba(124,58,237,0.10)", spin: true },
    completed: { label: "Done",      color: "#16A34A",                  bg: "rgba(22,163,74,0.12)" },
    error:     { label: "Error",     color: "var(--color-danger)",      bg: "rgba(220,38,38,0.10)" },
  };
  const s = map[status];
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: "4px",
      padding: "3px 10px", borderRadius: "var(--radius-full)",
      fontSize: "0.75rem", fontWeight: 600,
      color: s.color, background: s.bg,
    }}>
      {s.spin && <Loader2 style={{ width: 11, height: 11, animation: "spin 1s linear infinite" }} />}
      {status === "completed" && <CheckCircle2 style={{ width: 11, height: 11 }} />}
      {status === "error"     && <AlertCircle  style={{ width: 11, height: 11 }} />}
      {s.label}
    </span>
  );
};

export default function TestDashboardPage() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [globalEverySec, setGlobalEverySec] = useState<number>(1.0);
  const [globalWithCaptions, setGlobalWithCaptions] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState("videos");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newVideos: VideoItem[] = Array.from(e.target.files).map((file) => ({
        id: crypto.randomUUID(), file, cameraId: 99, status: 'pending',
      }));
      setVideos((prev) => [...prev, ...newVideos]);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const removeVideo = (id: string) => setVideos((prev) => prev.filter((v) => v.id !== id));
  const updateVideoCameraId = (id: string, cameraId: number) =>
    setVideos((prev) => prev.map((v) => (v.id === id ? { ...v, cameraId } : v)));

  const startProcessing = async () => {
    setIsProcessing(true);
    const pendingVideos = videos.filter(v => v.status === 'pending' || v.status === 'error');
    for (const video of pendingVideos) {
      setVideos(prev => prev.map(v => v.id === video.id ? { ...v, status: 'uploading', error: undefined } : v));
      try {
        const res = await api.uploadVideo(video.file, { camera_id: video.cameraId, every_sec: globalEverySec, with_captions: globalWithCaptions });
        setVideos(prev => prev.map(v => v.id === video.id ? { ...v, status: 'indexing', result: res } : v));
        let frames: any[] = [];
        try { frames = await api.listClipFrames(res.clip_path, 1000, "yolo"); } catch {}
        setVideos(prev => prev.map(v => v.id === video.id ? { ...v, status: 'completed', result: res, frames: frames || [] } : v));
      } catch (err: any) {
        setVideos(prev => prev.map(v => v.id === video.id ? { ...v, status: 'error', error: err?.message || String(err) } : v));
      }
    }
    setIsProcessing(false);
    if (videos.some(v => v.status === 'completed')) setActiveTab("results");
  };

  const completedCount = videos.filter(v => v.status === 'completed').length;
  const canChat = completedCount > 0;

  return (
    <MainLayout>
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--gap-island)", height: "100%" }}>

        {/* Page header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <div style={{
              width: 32, height: 32, borderRadius: "var(--radius-md)",
              background: "var(--color-primary-light)", display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <FlaskConical style={{ width: 16, height: 16, color: "var(--color-primary)" }} />
            </div>
            <div>
              <h1 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--color-text)", lineHeight: 1.2 }}>Test Dashboard</h1>
              <p style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: 1 }}>Batch process videos and query results</p>
            </div>
          </div>
          <a
            href={`${API_BASE.replace(/\/+$/, "")}/docs`}
            target="_blank"
            rel="noreferrer"
            style={{ fontSize: "0.8125rem", color: "var(--color-primary)", textDecoration: "none", fontWeight: 500 }}
            onMouseEnter={e => (e.currentTarget.style.textDecoration = "underline")}
            onMouseLeave={e => (e.currentTarget.style.textDecoration = "none")}
          >
            Backend Docs →
          </a>
        </div>

        {/* Tab bar */}
        <div style={{
          display: "flex", gap: "4px",
          background: "var(--color-surface)", border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-lg)", padding: "4px", boxShadow: "var(--shadow-sm)",
          flexShrink: 0,
        }}>
          {TABS.map(({ id, label, icon: Icon }) => {
            const active = activeTab === id;
            const disabled = id === "conversation" && !canChat;
            return (
              <button
                key={id}
                disabled={disabled}
                onClick={() => !disabled && setActiveTab(id)}
                style={{
                  flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: "6px",
                  padding: "8px 12px", borderRadius: "var(--radius-md)", border: "none",
                  fontSize: "0.8125rem", fontWeight: active ? 600 : 500, cursor: disabled ? "not-allowed" : "pointer",
                  background: active ? "var(--color-primary-light)" : "transparent",
                  color: active ? "var(--color-primary)" : disabled ? "var(--color-text-faint)" : "var(--color-text-muted)",
                  transition: "all 150ms ease",
                  fontFamily: "var(--font-ui)",
                }}
                className="test-tab-btn"
                data-active={active ? "true" : undefined}
              >
                <Icon style={{ width: 14, height: 14 }} />
                {label}
                {id === "videos" && videos.length > 0 && (
                  <span style={{
                    background: "var(--color-primary)", color: "#fff",
                    borderRadius: "var(--radius-full)", fontSize: "0.6875rem", fontWeight: 700,
                    padding: "0 6px", lineHeight: "18px", minWidth: 18, textAlign: "center",
                  }}>{videos.length}</span>
                )}
                {id === "results" && completedCount > 0 && (
                  <span style={{
                    background: "var(--color-primary)", color: "#fff",
                    borderRadius: "var(--radius-full)", fontSize: "0.6875rem", fontWeight: 700,
                    padding: "0 6px", lineHeight: "18px", minWidth: 18, textAlign: "center",
                  }}>{completedCount}</span>
                )}
              </button>
            );
          })}
        </div>

        {/* Tab: Videos */}
        {activeTab === "videos" && (
          <div style={{
            flex: 1, display: "flex", flexDirection: "column", gap: "var(--gap-inner)",
            background: "var(--color-surface)", border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-lg)", boxShadow: "var(--shadow-sm)",
            padding: "16px", overflow: "hidden", minHeight: 0,
          }}>
            {/* Settings row */}
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: "12px", alignItems: "end",
              background: "var(--color-surface-raised)", border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-md)", padding: "12px 14px", flexShrink: 0,
            }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                  Sample every (sec)
                </label>
                <input
                  type="number" step="0.1" min={0.1} value={globalEverySec}
                  onChange={(e) => setGlobalEverySec(parseFloat(e.target.value || "1.0"))}
                  style={{
                    padding: "7px 10px", border: "1px solid var(--color-border)",
                    borderRadius: "var(--radius-md)", background: "var(--color-surface)",
                    color: "var(--color-text)", fontSize: "0.875rem", fontFamily: "var(--font-ui)",
                    outline: "none", transition: "border-color 150ms",
                  }}
                  onFocus={e => (e.target.style.borderColor = "var(--color-primary)")}
                  onBlur={e => (e.target.style.borderColor = "var(--color-border)")}
                />
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer", paddingBottom: "8px" }}>
                <input
                  type="checkbox" checked={globalWithCaptions}
                  onChange={(e) => setGlobalWithCaptions(e.target.checked)}
                  style={{ width: 15, height: 15, accentColor: "var(--color-primary)", cursor: "pointer" }}
                />
                <span style={{ fontSize: "0.875rem", color: "var(--color-text)", fontWeight: 500 }}>Generate captions</span>
              </label>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isProcessing}
                className="add-videos-btn"
                style={{
                  padding: "8px 16px", borderRadius: "var(--radius-md)",
                  border: "1.5px solid var(--color-border)", background: "var(--color-surface)",
                  color: "var(--color-text)", fontSize: "0.8125rem", fontWeight: 600,
                  cursor: isProcessing ? "not-allowed" : "pointer",
                  opacity: isProcessing ? 0.5 : 1, transition: "all 150ms", fontFamily: "var(--font-ui)",
                }}
              >
                + Add Videos
              </button>
              <input type="file" multiple accept="video/mp4" ref={fileInputRef} style={{ display: "none" }} onChange={handleFileSelect} />
            </div>

            {/* Video list */}
            <ScrollArea style={{ flex: 1, minHeight: 0 }}>
              {videos.length === 0 ? (
                <div style={{ padding: "48px 16px", textAlign: "center", color: "var(--color-text-faint)" }}>
                  <Video style={{ width: 36, height: 36, margin: "0 auto 10px", opacity: 0.3 }} />
                  <p style={{ fontSize: "0.875rem" }}>No videos added yet</p>
                </div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: "6px", padding: "2px 2px" }}>
                  {videos.map((video) => (
                    <div key={video.id} style={{
                      display: "flex", alignItems: "center", gap: "12px",
                      padding: "10px 12px", borderRadius: "var(--radius-md)",
                      border: "1px solid var(--color-border)", background: "var(--color-surface-raised)",
                    }}>
                      <div style={{
                        width: 36, height: 36, borderRadius: "var(--radius-sm)",
                        background: "var(--color-surface)", border: "1px solid var(--color-border)",
                        display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0,
                      }}>
                        <FileVideo style={{ width: 16, height: 16, color: "var(--color-text-faint)" }} />
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <p style={{ fontSize: "0.8125rem", fontWeight: 600, color: "var(--color-text)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={video.file.name}>
                          {video.file.name}
                        </p>
                        <p style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: 1 }}>
                          {Math.round(video.file.size / 1024)} KB
                        </p>
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: "12px", flexShrink: 0 }}>
                        <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
                          <label style={{ fontSize: "0.625rem", fontWeight: 700, color: "var(--color-text-faint)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Camera ID</label>
                          <input
                            type="number" min={0} value={video.cameraId}
                            onChange={(e) => updateVideoCameraId(video.id, parseInt(e.target.value || "0", 10))}
                            disabled={video.status !== 'pending' && video.status !== 'error'}
                            style={{
                              width: 72, padding: "5px 8px", fontSize: "0.8125rem",
                              border: "1px solid var(--color-border)", borderRadius: "var(--radius-sm)",
                              background: "var(--color-surface)", color: "var(--color-text)",
                              fontFamily: "var(--font-ui)", outline: "none",
                              opacity: (video.status !== 'pending' && video.status !== 'error') ? 0.5 : 1,
                            }}
                          />
                        </div>
                        <div style={{ width: 90, display: "flex", justifyContent: "center" }}>
                          {statusPill(video.status)}
                        </div>
                        <button
                          onClick={() => removeVideo(video.id)}
                          disabled={isProcessing && video.status !== 'pending' && video.status !== 'error'}
                          className="delete-video-btn"
                          style={{
                            width: 30, height: 30, borderRadius: "var(--radius-sm)",
                            border: "none", background: "transparent",
                            color: "var(--color-text-faint)", cursor: "pointer",
                            display: "flex", alignItems: "center", justifyContent: "center",
                            transition: "all 150ms",
                          }}
                        >
                          <Trash2 style={{ width: 14, height: 14 }} />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </ScrollArea>

            {/* Start button */}
            <div style={{ flexShrink: 0, display: "flex", justifyContent: "flex-end" }}>
              <button
                onClick={startProcessing}
                disabled={isProcessing || videos.filter(v => v.status === 'pending' || v.status === 'error').length === 0}
                style={{
                  padding: "9px 22px", borderRadius: "var(--radius-md)", border: "none",
                  background: "var(--color-primary)", color: "#fff",
                  fontSize: "0.875rem", fontWeight: 600, cursor: "pointer",
                  display: "flex", alignItems: "center", gap: "8px",
                  opacity: (isProcessing || videos.filter(v => v.status === 'pending' || v.status === 'error').length === 0) ? 0.5 : 1,
                  transition: "opacity 150ms", fontFamily: "var(--font-ui)",
                }}
              >
                {isProcessing && <Loader2 style={{ width: 14, height: 14, animation: "spin 1s linear infinite" }} />}
                {isProcessing ? "Processing Queue…" : "Start Indexing Queue"}
              </button>
            </div>
          </div>
        )}

        {/* Tab: Results */}
        {activeTab === "results" && (
          <ScrollArea style={{ flex: 1, minHeight: 0 }}>
            {videos.filter(v => v.status === 'completed').length === 0 ? (
              <div style={{ padding: "64px 16px", textAlign: "center", color: "var(--color-text-faint)" }}>
                <FileVideo style={{ width: 40, height: 40, margin: "0 auto 12px", opacity: 0.2 }} />
                <p style={{ fontSize: "0.875rem" }}>No indexing results yet.</p>
                <p style={{ fontSize: "0.8125rem", marginTop: 4, color: "var(--color-text-faint)" }}>Process some videos to see results here.</p>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "12px", paddingBottom: "24px" }}>
                {videos.filter(v => v.status === 'completed').map((video) => (
                  <div key={video.id} style={{
                    background: "var(--color-surface)", border: "1px solid var(--color-border)",
                    borderRadius: "var(--radius-lg)", boxShadow: "var(--shadow-sm)",
                    overflow: "hidden",
                  }}>
                    {/* card header */}
                    <div style={{ padding: "14px 16px", borderBottom: "1px solid var(--color-border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                      <div>
                        <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-text)", wordBreak: "break-all" }}>{video.file.name}</p>
                        <p style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: 2 }}>Camera ID: {video.cameraId}</p>
                      </div>
                      <span style={{
                        padding: "3px 10px", borderRadius: "var(--radius-full)",
                        fontSize: "0.75rem", fontWeight: 600,
                        color: "var(--color-success)", background: "rgba(22,163,74,0.10)",
                      }}>Indexed</span>
                    </div>
                    {/* card body */}
                    <div style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: "12px" }}>
                      {/* Stats */}
                      <div style={{
                        display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "12px",
                        background: "var(--color-surface-raised)", borderRadius: "var(--radius-md)", padding: "12px",
                      }}>
                        {[
                          { label: "Clip Path", value: <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", wordBreak: "break-all" }}>{video.result?.clip_path}</span> },
                          { label: "Clip URL", value: video.result?.clip_url ? (
                            <a href={API_BASE.replace(/\/+$/, "") + video.result.clip_url} target="_blank" rel="noreferrer"
                              style={{ color: "var(--color-primary)", fontSize: "0.8125rem", textDecoration: "none", fontWeight: 500 }}>Open Video</a>
                          ) : "N/A" },
                          { label: "Frames", value: <span style={{ fontWeight: 600, color: "var(--color-text)" }}>{video.frames?.length || 0}</span> },
                          { label: "Raw Data", value: <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8125rem" }}>{Object.keys(video.result?.indexing || {}).length} keys</span> },
                        ].map(({ label, value }) => (
                          <div key={label}>
                            <p style={{ fontSize: "0.6875rem", color: "var(--color-text-faint)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>{label}</p>
                            <div style={{ fontSize: "0.8125rem", color: "var(--color-text-muted)" }}>{value}</div>
                          </div>
                        ))}
                      </div>
                      {/* Frames table */}
                      {video.frames && video.frames.length > 0 && (
                        <div style={{ border: "1px solid var(--color-border)", borderRadius: "var(--radius-md)", overflow: "hidden" }}>
                          <div style={{ maxHeight: 300, overflow: "auto" }}>
                            <table style={{ width: "100%", fontSize: "0.75rem", borderCollapse: "collapse" }}>
                              <thead>
                                <tr style={{ background: "var(--color-surface-raised)" }}>
                                  {["Idx", "Time", "Caption", "Objects"].map(h => (
                                    <th key={h} style={{ padding: "8px 10px", fontWeight: 600, color: "var(--color-text-muted)", textAlign: "left", position: "sticky", top: 0, background: "var(--color-surface-raised)", borderBottom: "1px solid var(--color-border)" }}>{h}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {video.frames.slice(0, 50).map((fr, i) => (
                                  <tr key={i} style={{ borderBottom: "1px solid var(--color-border)" }} className="result-row">
                                    <td style={{ padding: "7px 10px", color: "var(--color-text-muted)" }}>{fr.frame_index}</td>
                                    <td style={{ padding: "7px 10px", color: "var(--color-text-muted)" }}>{fr.frame_ts}</td>
                                    <td style={{ padding: "7px 10px", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--color-text)" }} title={fr.caption}>{fr.caption || "-"}</td>
                                    <td style={{ padding: "7px 10px", maxWidth: 150, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "var(--color-text-muted)" }}>
                                      {Array.isArray((fr as any).object_captions) ? ((fr as any).object_captions as string[]).join(", ") : "-"}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          {video.frames.length > 50 && (
                            <div style={{ padding: "8px", textAlign: "center", fontSize: "0.75rem", color: "var(--color-text-faint)", background: "var(--color-surface-raised)", borderTop: "1px solid var(--color-border)" }}>
                              Showing first 50 of {video.frames.length} frames
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </ScrollArea>
        )}

        {/* Tab: Conversation */}
        {activeTab === "conversation" && (
          <div style={{
            flex: 1, minHeight: 0,
            background: "var(--color-surface)", border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-lg)", boxShadow: "var(--shadow-sm)", overflow: "hidden",
            display: "flex", flexDirection: "column",
          }}>
            {canChat ? (
              <ChatInterface onShowSteps={() => {}} />
            ) : (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "var(--color-text-faint)" }}>
                <MessageSquare style={{ width: 40, height: 40, marginBottom: 12, opacity: 0.2 }} />
                <p style={{ fontSize: "0.875rem" }}>Conversation unavailable</p>
                <p style={{ fontSize: "0.8125rem", marginTop: 4 }}>Index at least one video to start chatting.</p>
              </div>
            )}
          </div>
        )}

      </div>

      <style>{`
        .test-tab-btn:hover:not([data-active]):not(:disabled) {
          background: var(--color-surface-raised) !important;
          color: var(--color-text) !important;
        }
        .add-videos-btn:hover:not(:disabled) {
          background: var(--color-surface-raised) !important;
          border-color: var(--color-border-strong) !important;
        }
        .delete-video-btn:hover:not(:disabled) {
          color: var(--color-danger) !important;
          background: rgba(220,38,38,0.08) !important;
        }
        .result-row:hover { background: var(--color-surface-raised); }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </MainLayout>
  );
}
