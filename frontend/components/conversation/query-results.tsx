"use client"

import { useState, useEffect } from "react"
import { createPortal } from "react-dom"
import { Play, X } from "lucide-react"
import { API_BASE, type ChatSendResponse } from "@/lib/api"

interface QueryResultsProps {
  onShowSteps: () => void
  response: ChatSendResponse | null
}

export function QueryResults({ onShowSteps, response }: QueryResultsProps) {
  const [selectedVideoUrl, setSelectedVideoUrl] = useState<string | null>(null)
  const [showVideoModal, setShowVideoModal] = useState(false)
  const [mounted, setMounted] = useState(false)

  const combined = (response?.combined_tracks as any[]) || []
  const merged = (response?.merged_tracks as any[]) || []
  const raw = (response?.results as any[]) || []
  const sem = (response?.semantic_results as any[]) || []

  let view: any[] = []
  let title = "Detections"
  if (sem.length > 0 && (response?.mode === "unstructured" || response?.mode === "hybrid")) {
    view = sem
    title = response?.mode === "hybrid" ? "Hybrid Results" : "Semantic Results"
  } else {
    view = combined.length > 0 ? combined : (merged.length > 0 ? merged : raw)
    title = combined.length > 0 ? "Combined Clips" : merged.length > 0 ? "Merged Segments" : "Detections"
  }
  const count = view.length

  // Set mounted state for portal
  useEffect(() => {
    setMounted(true)
  }, [])

  // Close modal on ESC key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape" && showVideoModal) {
        setShowVideoModal(false)
        setSelectedVideoUrl(null)
      }
    }
    window.addEventListener("keydown", handleEsc)
    return () => window.removeEventListener("keydown", handleEsc)
  }, [showVideoModal])

  const handleVideoClick = (clipUrl: string) => {
    setSelectedVideoUrl(`${API_BASE}${clipUrl}`)
    setShowVideoModal(true)
  }

  const closeModal = () => {
    setShowVideoModal(false)
    setSelectedVideoUrl(null)
  }

  return (
    <div className="glass-card glass-noise rounded-2xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-foreground">{title} ({count} results)</h3>
        <button onClick={onShowSteps} className="text-xs link">
          View AI Steps
        </button>
      </div>

      <div className="grid grid-cols-5 gap-3">
        {view.map((item, i) => (
          <div
            key={i}
            className="group relative overflow-hidden rounded-lg aspect-video cursor-pointer border border-[color:var(--border)] bg-[color-mix(in oklab,var(--foreground) 6%,transparent)]"
            onClick={() => (item as any).clip_url && handleVideoClick((item as any).clip_url)}
          >
            {(item as any).clip_url ? (
              <video
                key={(item as any).clip_url}
                src={`${API_BASE}${(item as any).clip_url}`}
                className="absolute inset-0 w-full h-full object-cover"
                muted
                loop
                playsInline
                preload="metadata"
                crossOrigin="anonymous"
                onError={(e) => {
                  const el = e.currentTarget as HTMLVideoElement;
                  const parent = el.parentElement;
                  if (parent) {
                    const url = `${API_BASE}${(item as any).clip_url}`;
                    parent.innerHTML = `
                      <div class="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center p-4 text-center">
                        <div class="text-xs text-white/80">
                          Unable to preview video in the browser. 
                          <a href="${url}" class="text-cyan-300 underline" target="_blank" rel="noreferrer">Download/Open Clip</a>
                        </div>
                      </div>
                    `;
                  }
                }}
              />
            ) : (
              <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
                <div className="w-10 h-10 rounded-full bg-[var(--btn-primary)] hover:bg-[var(--btn-primary-hover)] text-white flex items-center justify-center shadow transition-colors">
                  <Play className="w-5 h-5" />
                </div>
              </div>
            )}

            <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />

            {/* Object Badges - Always visible at top */}
            {(item as any).objects && (item as any).objects.length > 0 && (
              <div className="absolute top-2 left-2 right-2 flex flex-wrap gap-1 pointer-events-none z-10">
                {(item as any).objects.slice(0, 2).map((obj: any, idx: number) => (
                  <span
                    key={idx}
                    className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-cyan-500/90 text-white shadow-lg backdrop-blur-sm"
                  >
                    {obj.object_name}
                    {obj.color && ` • ${obj.color}`}
                    {obj.confidence != null && ` • ${Math.round(obj.confidence * 100)}%`}
                  </span>
                ))}
                {(item as any).objects.length > 2 && (
                  <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-purple-500/90 text-white shadow-lg">
                    +{(item as any).objects.length - 2} more
                  </span>
                )}
              </div>
            )}

            <div className="absolute bottom-0 left-0 right-0 p-2 translate-y-full group-hover:translate-y-0 transition-transform">
              <p className="text-xs font-semibold text-white">
                {/* Prefer semantic frame timestamp if present */}
                {(item as any).frames?.[0]?.frame_ts
                  ? new Date((item as any).frames?.[0]?.frame_ts).toLocaleString()
                  : (item as any).start && (item as any).end
                    ? `${new Date((item as any).start).toLocaleTimeString()} - ${new Date((item as any).end).toLocaleTimeString()}`
                    : (item as any).timestamp
                      ? new Date((item as any).timestamp).toLocaleString()
                      : ""}
              </p>
              {/* Semantic score shown when available */}
              {(item as any).score_norm != null && (
                <p className="text-xs text-cyan-400">score {(item as any).score_norm.toFixed(2)}</p>
              )}
              {/* Structured durations */}
              {(item as any).duration_seconds != null && (
                <p className="text-xs text-cyan-400">{(item as any).duration_seconds}s span</p>
              )}
              {/* All detected objects details */}
              {(item as any).objects && (item as any).objects.length > 0 && (
                <div className="mt-1 space-y-0.5">
                  <p className="text-[10px] text-white/70 font-medium">Detected Objects:</p>
                  {(item as any).objects.map((obj: any, idx: number) => (
                    <div key={idx} className="text-[10px] text-white/90 flex items-center gap-1">
                      <span className="font-medium">{obj.object_name}</span>
                      {obj.color && <span className="text-cyan-300">• {obj.color}</span>}
                      {obj.confidence != null && (
                        <span className="text-green-300">• {Math.round(obj.confidence * 100)}%</span>
                      )}
                      {obj.track_id != null && (
                        <span className="text-purple-300">• ID:{obj.track_id}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
              {/* Raw detection confidence fallback */}
              {!(item as any).duration_seconds && (item as any).objects && (
                <p className="text-xs text-cyan-400">
                  {Math.round((((item as any).objects?.[0]?.confidence ?? 0) * 100))}% confidence
                </p>
              )}
            </div>
          </div>
        ))}
      </div>

      <button className="w-full rounded-2xl py-2 text-sm bg-[var(--btn-primary)] text-white hover:bg-[var(--btn-primary-hover)] transition-colors">
        View All / Download
      </button>

      {/* Video Modal - rendered as portal to escape stacking context */}
      {mounted && showVideoModal && selectedVideoUrl && createPortal(
        <div 
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/80 dark:bg-black/90 backdrop-blur-sm"
          onClick={closeModal}
        >
          <div 
            className="relative w-[90vw] max-w-5xl bg-white dark:bg-gray-900 rounded-2xl shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close button */}
            <button
              onClick={closeModal}
              className="absolute top-4 right-4 z-10 p-2 rounded-full bg-black/50 dark:bg-white/10 hover:bg-black/70 dark:hover:bg-white/20 text-white transition-colors"
              aria-label="Close"
            >
              <X className="w-6 h-6" />
            </button>

            {/* Video player */}
            <video
              src={selectedVideoUrl}
              className="w-full max-h-[85vh] bg-black"
              controls
              autoPlay
              playsInline
              crossOrigin="anonymous"
            />
          </div>
        </div>,
        document.body
      )}
    </div>
  )
}
