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
  const [selectedItem, setSelectedItem] = useState<any>(null)

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

  // Close modal on ESC key
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape" && showVideoModal) {
        setShowVideoModal(false)
        setSelectedVideoUrl(null)
        setSelectedItem(null)
      }
    }
    window.addEventListener("keydown", handleEsc)
    return () => window.removeEventListener("keydown", handleEsc)
  }, [showVideoModal])

  const handleVideoClick = (clipUrl: string, item: any) => {
    setSelectedVideoUrl(`${API_BASE}${clipUrl}`)
    setSelectedItem(item)
    setShowVideoModal(true)
  }

  const closeModal = () => {
    setShowVideoModal(false)
    setSelectedVideoUrl(null)
    setSelectedItem(null)
  }

  return (
    <div className="glass-card glass-noise rounded-2xl p-6 space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-bold text-foreground">{title}</h3>
          <p className="text-sm text-foreground/60 mt-0.5">{count} {count === 1 ? 'result' : 'results'} found</p>
        </div>
        <button 
          onClick={onShowSteps} 
          className="text-sm px-4 py-2 rounded-lg bg-[color-mix(in_oklab,var(--foreground)_8%,transparent)] hover:bg-[color-mix(in_oklab,var(--foreground)_12%,transparent)] transition-colors font-medium text-foreground/80 hover:text-foreground"
        >
          View AI Steps
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {view.map((item, i) => (
          <div
            key={i}
            className="group relative overflow-hidden rounded-xl aspect-video cursor-pointer bg-gradient-to-br from-slate-900 to-slate-800 shadow-xl hover:shadow-2xl transition-all duration-300 hover:scale-[1.03] ring-1 ring-white/10 hover:ring-cyan-500"
            onClick={() => {
              if ((item as any).clip_url) {
                handleVideoClick((item as any).clip_url, item)
              }
            }}
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
              />
            ) : (
              <div className="absolute inset-0 bg-gradient-to-br from-slate-800 to-slate-900 flex items-center justify-center">
                <div className="text-sm text-white/40">No preview</div>
              </div>
            )}

            {/* Small index number */}
            <div className="absolute top-2 left-2 w-6 h-6 rounded-full bg-black/60 backdrop-blur-sm flex items-center justify-center">
              <span className="text-[10px] font-bold text-white">{i + 1}</span>
            </div>
          </div>
        ))}
      </div>

      <button className="w-full rounded-xl py-3 text-sm font-semibold bg-gradient-to-r from-[var(--btn-primary)] to-blue-600 text-white hover:shadow-lg hover:scale-[1.01] transition-all duration-200 flex items-center justify-center gap-2">
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
        View All / Download
      </button>

      {/* Enhanced Video Modal with Details */}
      {showVideoModal && selectedVideoUrl && selectedItem && typeof document !== 'undefined' && createPortal(
        <div 
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/90 backdrop-blur-sm p-4"
          onClick={closeModal}
        >
          <div 
            className="relative w-full max-w-7xl bg-gradient-to-br from-slate-900 to-slate-800 rounded-2xl shadow-2xl overflow-hidden border border-white/10"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Close button */}
            <button
              onClick={closeModal}
              className="absolute top-4 right-4 z-20 p-2.5 rounded-full bg-black/70 hover:bg-black/90 text-white transition-all duration-200 hover:scale-110 backdrop-blur-sm border border-white/10"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="grid lg:grid-cols-3 gap-0">
              {/* Video Player - 2/3 width on large screens */}
              <div className="lg:col-span-2 bg-black relative">
                <video
                  src={selectedVideoUrl}
                  className="w-full h-full max-h-[85vh] object-contain"
                  controls
                  autoPlay
                  playsInline
                  crossOrigin="anonymous"
                />
              </div>

              {/* Details Panel - 1/3 width on large screens */}
              <div className="p-6 space-y-5 bg-slate-900/80 backdrop-blur-sm max-h-[85vh] overflow-y-auto">
                {/* Header */}
                <div className="pb-4 border-b border-white/10">
                  <h3 className="text-xl font-bold text-white mb-1">Clip Details</h3>
                  <p className="text-xs text-white/40 uppercase tracking-wider">Detection Information</p>
                </div>

                {/* Timestamp */}
                {(selectedItem.timestamp || selectedItem.start) && (
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-cyan-400">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      <span className="text-xs font-bold uppercase tracking-wide">Timestamp</span>
                    </div>
                    <p className="text-sm text-white/90 pl-6 font-medium">
                      {selectedItem.frames?.[0]?.frame_ts
                        ? new Date(selectedItem.frames[0].frame_ts).toLocaleString()
                        : selectedItem.start && selectedItem.end
                          ? `${new Date(selectedItem.start).toLocaleString()} - ${new Date(selectedItem.end).toLocaleTimeString()}`
                          : selectedItem.timestamp
                            ? new Date(selectedItem.timestamp).toLocaleString()
                            : ""}
                    </p>
                  </div>
                )}

                {/* Duration & Score */}
                <div className="grid grid-cols-2 gap-3">
                  {selectedItem.duration_seconds != null && (
                    <div className="px-4 py-3 rounded-xl bg-gradient-to-br from-blue-500/20 to-cyan-500/10 border border-blue-500/30">
                      <p className="text-[10px] text-blue-300 font-bold uppercase tracking-wide">Duration</p>
                      <p className="text-2xl font-bold text-white mt-1">{selectedItem.duration_seconds}s</p>
                    </div>
                  )}
                  {selectedItem.score_norm != null && (
                    <div className="px-4 py-3 rounded-xl bg-gradient-to-br from-purple-500/20 to-pink-500/10 border border-purple-500/30">
                      <p className="text-[10px] text-purple-300 font-bold uppercase tracking-wide">Score</p>
                      <p className="text-2xl font-bold text-white mt-1">{selectedItem.score_norm.toFixed(2)}</p>
                    </div>
                  )}
                </div>

                {/* Detected Objects */}
                {selectedItem.objects && selectedItem.objects.length > 0 && (
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 text-emerald-400">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
                      </svg>
                      <span className="text-xs font-bold uppercase tracking-wide">Detected Objects</span>
                    </div>
                    <div className="space-y-3">
                      {selectedItem.objects.map((obj: any, idx: number) => (
                        <div key={idx} className="p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-cyan-500/30 transition-all duration-200">
                          <div className="flex items-center justify-between mb-3">
                            <span className="text-base font-bold text-white">{obj.object_name}</span>
                            {obj.confidence != null && (
                              <span className="px-3 py-1 rounded-full bg-gradient-to-r from-green-500/30 to-emerald-500/30 text-green-300 text-sm font-bold border border-green-500/20">
                                {Math.round(obj.confidence * 100)}%
                              </span>
                            )}
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {obj.color && (
                              <div className="px-3 py-1.5 rounded-lg bg-cyan-500/20 border border-cyan-500/30">
                                <span className="text-xs text-cyan-300 font-semibold">Color: {obj.color}</span>
                              </div>
                            )}
                            {obj.track_id != null && (
                              <div className="px-3 py-1.5 rounded-lg bg-purple-500/20 border border-purple-500/30">
                                <span className="text-xs text-purple-300 font-semibold">Track ID: {obj.track_id}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Camera Info */}
                {selectedItem.camera_id != null && (
                  <div className="pt-4 border-t border-white/10">
                    <div className="flex items-center gap-2 text-amber-400 mb-2">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      <span className="text-xs font-bold uppercase tracking-wide">Camera</span>
                    </div>
                    <p className="text-sm text-white/90 font-medium pl-6">Camera {selectedItem.camera_id}</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  )
}
