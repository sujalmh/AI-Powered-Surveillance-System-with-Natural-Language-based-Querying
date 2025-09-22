"use client"

import { useState, useEffect, useRef } from "react"
import { X, Play, Pause, Volume2, VolumeX, Maximize2, Minimize2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent } from "@/components/ui/dialog"

interface VideoModalProps {
  isOpen: boolean
  onClose: () => void
  onActivate?: () => void
  video: {
    id: string
    title: string
    timestamp: string
    location: string
    active: boolean // Added to track camera active status
  }
}

export function VideoModal({ isOpen, onClose, video }: VideoModalProps) {
  const [isPlaying, setIsPlaying] = useState(true)
  const [isMuted, setIsMuted] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [streamError, setStreamError] = useState<string | null>(null)
  const [isCameraActive, setIsCameraActive] = useState(video.active)
  const [activationError, setActivationError] = useState<string | null>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  const streamUrl = `http://localhost:5000/cam/stream/${video.id}`

  // Handle play/pause by toggling the stream source
  useEffect(() => {
    if (!isOpen) {
      setIsPlaying(true) // Reset to playing when modal closes
      setStreamError(null)
      setActivationError(null)
      if (imgRef.current) imgRef.current.src = "" // Ensure stream stops
      return
    }

    if (imgRef.current) {
      if (isPlaying && isCameraActive) {
        imgRef.current.src = streamUrl
      } else {
        imgRef.current.src = "" // Stop fetching stream
      }
    }
  }, [isPlaying, isOpen, streamUrl, isCameraActive])

  // Sync isCameraActive with video.active when prop changes
  useEffect(() => {
    setIsCameraActive(video.active)
  }, [video.active])

  // Handle fullscreen state changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener("fullscreenchange", handleFullscreenChange)
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange)
  }, [])

  const togglePlay = () => {
    setIsPlaying(!isPlaying)
    setStreamError(null) // Reset stream error when attempting to play
  }

  const toggleMute = () => setIsMuted(!isMuted)

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch((err) => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`)
      })
    } else if (document.exitFullscreen) {
      document.exitFullscreen()
    }
  }

  const handleStreamError = () => {
    setStreamError("Failed to load stream. Camera or video may be unavailable.")
    setIsPlaying(false)
  }

  const activateCamera = async () => {
    try {
      setActivationError(null)
      const response = await fetch(`http://localhost:5000/cam/activate/${video.id}`, {
        method: "POST", // Assuming POST for activation
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to activate camera")
      }
      setIsCameraActive(true) // Optimistic update
      onActivate?.()
    } catch (error) {
      console.error("Error activating camera:", error)
      setActivationError(error instanceof Error ? error.message : "Failed to activate camera")
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-6xl p-0 overflow-hidden bg-black">
        <div className="relative">
          {/* Video content */}
          <div className="aspect-video bg-black flex items-center justify-center">
            {!isCameraActive ? (
              <div className="text-white text-center">
                <p>Camera is offline</p>
                <Button
                  variant="outline"
                  className="mt-2"
                  onClick={activateCamera}
                  disabled={activationError !== null}
                >
                  Activate Camera
                </Button>
                {activationError && (
                  <p className="text-red-500 text-xs mt-2">{activationError}</p>
                )}
              </div>
            ) : streamError ? (
              <div className="text-white text-center">
                <p>{streamError}</p>
                <Button
                  variant="outline"
                  className="mt-2"
                  onClick={() => {
                    setIsPlaying(true)
                    setStreamError(null)
                  }}
                >
                  Retry
                </Button>
              </div>
            ) : !isPlaying ? (
              <div className="text-white text-center">
                <p>Stream Paused</p>
                <Button variant="outline" className="mt-2" onClick={togglePlay}>
                  Resume
                </Button>
              </div>
            ) : (
              <img
                ref={imgRef}
                src={isPlaying && isCameraActive ? streamUrl : ""}
                alt={video.title}
                className="h-full w-full object-cover"
                onError={handleStreamError}
              />
            )}
          </div>

          {/* Top controls */}
          <div className="absolute top-0 left-0 right-0 p-4 flex justify-between items-center bg-gradient-to-b from-black/80 to-transparent">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="bg-primary/20 text-primary-foreground border-none">
                {video.timestamp}
              </Badge>
              <span className="text-white text-sm">{video.location}</span>
            </div>
            <Button variant="ghost" size="icon" className="text-white hover:bg-black/50" onClick={onClose}>
              <X className="h-5 w-5" />
            </Button>
          </div>

          {/* Bottom controls */}
          <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/80 to-transparent">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="icon" className="text-white hover:bg-black/50" onClick={togglePlay}>
                  {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                </Button>
                <Button variant="ghost" size="icon" className="text-white hover:bg-black/50" onClick={toggleMute}>
                  {isMuted ? <VolumeX className="h-5 w-5" /> : <Volume2 className="h-5 w-5" />}
                </Button>
              </div>
              <div>
                <h3 className="text-white font-medium">{video.title}</h3>
              </div>
              <Button variant="ghost" size="icon" className="text-white hover:bg-black/50" onClick={toggleFullscreen}>
                {isFullscreen ? <Minimize2 className="h-5 w-5" /> : <Maximize2 className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}