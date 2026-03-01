"use client"

import React, { useState, useEffect, useRef } from "react"
import { createPortal } from "react-dom"
import { Play, X, Download } from "lucide-react"
import { API_BASE, type ChatSendResponse } from "@/lib/api"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

interface QueryResultsProps {
  onShowSteps: () => void
  response: ChatSendResponse | null
}

export const QueryResults = React.memo(({ onShowSteps, response }: QueryResultsProps) => {
  const [selectedVideoUrl, setSelectedVideoUrl] = useState<string | null>(null)
  const [showVideoModal, setShowVideoModal] = useState(false)
  const [selectedItem, setSelectedItem] = useState<any>(null)
  const [actualVideoDuration, setActualVideoDuration] = useState<number | null>(null)
  const [videoError, setVideoError] = useState<string | null>(null)
  const [thumbErrors, setThumbErrors] = useState<Set<number>>(new Set())
  const videoRef = useRef<HTMLVideoElement>(null)

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
    setActualVideoDuration(null) // Reset duration when opening modal
    setVideoError(null)
    setShowVideoModal(true)
  }

  const closeModal = () => {
    setShowVideoModal(false)
    setSelectedVideoUrl(null)
    setSelectedItem(null)
    setActualVideoDuration(null)
  }

  return (
    <Card className="rounded-xl border-border bg-card shadow-sm overflow-hidden mt-4">
      <CardHeader className="py-4 px-5 border-b border-border bg-muted/40">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base text-foreground">{title}</CardTitle>
            <p className="text-xs text-muted-foreground mt-0.5">{count} {count === 1 ? 'result' : 'results'} found</p>
          </div>
          <Button 
            variant="outline"
            size="sm"
            onClick={onShowSteps} 
          >
            View AI Steps
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-5">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4 mb-5">
          {view.map((item, i) => (
            <div
              key={i}
              className="group relative overflow-hidden rounded-lg aspect-video cursor-pointer bg-stone-900 shadow-sm border border-border hover:border-primary transition-all duration-300 hover:scale-[1.02]"
              onClick={() => {
                if ((item as any).clip_url) {
                  handleVideoClick((item as any).clip_url, item)
                }
              }}
            >
              {(item as any).clip_url && !thumbErrors.has(i) ? (
                <video
                  key={(item as any).clip_url}
                  src={`${API_BASE}${(item as any).clip_url}#t=0.001`}
                  className="absolute inset-0 w-full h-full object-cover"
                  muted
                  loop
                  playsInline
                  preload="metadata"
                  crossOrigin="anonymous"
                  onMouseOver={e => (e.target as HTMLVideoElement).play().catch(()=> {})}
                  onMouseOut={e => {
                    const v = e.target as HTMLVideoElement;
                    v.pause();
                    v.currentTime = 0;
                  }}
                  onError={() => setThumbErrors(prev => new Set(prev).add(i))}
                />
              ) : (
                <div className="absolute inset-0 bg-stone-800 flex items-center justify-center">
                  <div className="text-xs text-stone-500 font-medium">
                    {(item as any).clip_url ? "Video unavailable" : "No preview"}
                  </div>
                </div>
              )}

              <div className="absolute top-2 left-2 w-5 h-5 rounded bg-stone-900/80 flex items-center justify-center">
                <span className="text-[10px] font-bold text-white">{i + 1}</span>
              </div>
              
              <div className="absolute inset-0 bg-black/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <Play className="w-8 h-8 text-white drop-shadow-sm" />
              </div>
            </div>
          ))}
        </div>

        <Button className="w-full gap-2" variant="default">
          <Download className="w-4 h-4" />
          View All / Download
        </Button>

        {showVideoModal && selectedVideoUrl && selectedItem && typeof document !== 'undefined' && createPortal(
          <div 
            className="fixed inset-0 z-[9999] flex items-center justify-center bg-stone-950/95 p-4 md:p-8"
            onClick={closeModal}
          >
            <div 
              className="relative w-full max-w-6xl bg-white dark:bg-stone-900 rounded-2xl shadow-sm overflow-hidden border border-stone-200 dark:border-stone-800"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                onClick={closeModal}
                className="absolute top-4 right-4 z-20 p-2 rounded-full bg-black/50 hover:bg-black/70 text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>

              <div className="grid lg:grid-cols-3 gap-0 h-full max-h-[85vh]">
                <div className="lg:col-span-2 bg-black flex items-center justify-center">
                  {videoError ? (
                    <div className="flex flex-col items-center justify-center gap-3 p-12 text-center">
                      <div className="text-red-400 text-sm font-medium">Failed to load video</div>
                      <div className="text-stone-500 text-xs max-w-xs">{videoError}</div>
                      <button
                        onClick={() => { setVideoError(null); if (videoRef.current) videoRef.current.load(); }}
                        className="mt-2 px-3 py-1.5 rounded-md bg-stone-800 hover:bg-stone-700 text-xs text-stone-300 transition-colors"
                      >
                        Retry
                      </button>
                    </div>
                  ) : (
                    <video
                      ref={videoRef}
                      src={selectedVideoUrl}
                      className="w-full max-h-full object-contain"
                      controls
                      autoPlay
                      playsInline
                      crossOrigin="anonymous"
                      onLoadedMetadata={() => {
                        if (videoRef.current) {
                          setActualVideoDuration(videoRef.current.duration)
                        }
                      }}
                      onError={(e) => {
                        const v = e.target as HTMLVideoElement;
                        const code = v.error?.code;
                        const msg = code === 4 ? "Video format not supported by browser"
                          : code === 2 ? "Network error loading video"
                          : code === 3 ? "Video decoding failed"
                          : "Could not play this video clip";
                        setVideoError(msg);
                      }}
                    />
                  )}
                </div>

                <div className="p-6 bg-card max-h-[85vh] overflow-y-auto border-l border-border">
                  <div className="pb-4 border-b border-border">
                    <h3 className="text-lg font-bold text-foreground mb-1">Clip Details</h3>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Detection Information</p>
                  </div>

                  {(selectedItem.timestamp || selectedItem.start) && (
                    <div className="mt-5 space-y-2">
                      <p className="text-xs font-bold uppercase tracking-wide text-primary">Timestamp</p>
                      <p className="text-sm text-foreground font-medium">
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

                  <div className="grid grid-cols-2 gap-4 mt-5">
                    {(actualVideoDuration !== null || selectedItem.duration_seconds != null) && (
                      <div className="p-3 rounded-xl bg-muted border border-border">
                        <p className="text-[10px] text-muted-foreground font-bold uppercase tracking-wide">Duration</p>
                        <p className="text-xl font-bold text-foreground mt-1">
                          {actualVideoDuration !== null 
                            ? `${actualVideoDuration.toFixed(2)}s`
                            : `${selectedItem.duration_seconds}s`}
                        </p>
                      </div>
                    )}
                    {selectedItem.score_norm != null && (
                      <div className="p-3 rounded-xl bg-muted border border-border">
                        <p className="text-[10px] text-muted-foreground font-bold uppercase tracking-wide">Score</p>
                        <p className="text-xl font-bold text-foreground mt-1">{selectedItem.score_norm.toFixed(2)}</p>
                      </div>
                    )}
                  </div>

                  {selectedItem.objects && selectedItem.objects.length > 0 && (
                    <div className="mt-6 space-y-3">
                      <p className="text-xs font-bold uppercase tracking-wide text-emerald-500">Detected Objects</p>
                      <div className="space-y-3">
                        {selectedItem.objects.map((obj: any, idx: number) => (
                          <div key={idx} className="p-4 rounded-xl bg-muted/50 border border-border">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-sm font-bold text-foreground">{obj.object_name}</span>
                              {obj.confidence != null && (
                                <span className="px-2 py-0.5 rounded text-xs font-bold bg-emerald-500/10 text-emerald-500">
                                  {Math.round(obj.confidence * 100)}%
                                </span>
                              )}
                            </div>
                            <div className="flex flex-wrap gap-2 mt-2">
                              {obj.color && (
                                <span className="px-2 py-1 rounded bg-muted border border-border text-xs font-medium text-muted-foreground">
                                  Color: {obj.color}
                                </span>
                              )}
                              {obj.track_id != null && (
                                <span className="px-2 py-1 rounded bg-muted border border-border text-xs font-medium text-muted-foreground">
                                  Track ID: {obj.track_id}
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {selectedItem.camera_id != null && (
                    <div className="mt-6 pt-4 border-t border-border">
                      <p className="text-xs font-bold uppercase tracking-wide text-amber-500">Camera</p>
                      <p className="text-sm text-foreground font-medium mt-1">Camera {selectedItem.camera_id}</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>,
          document.body
        )}
      </CardContent>
    </Card>
  )
})
QueryResults.displayName = "QueryResults";
