"use client"

import { useState, useEffect, useRef } from "react"
import { Play, Pause, Maximize2, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

interface CameraPreviewProps {
  id: string // Added to access camera_id for streaming
  name: string
  location: string
  active?: boolean
  hasAlert?: boolean
}

export function CameraPreview({ id, name, location, active = true, hasAlert = false }: CameraPreviewProps) {
  const [isPlaying, setIsPlaying] = useState(true)
  const [currentTime, setCurrentTime] = useState(new Date())
  const [streamError, setStreamError] = useState<string | null>(null)
  const imgRef = useRef<HTMLImageElement>(null)

  const streamUrl = `http://localhost:5000/cam/stream/${id}`

  // Update timestamp every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
    }, 1000)

    return () => clearInterval(timer)
  }, [])

  // Handle play/pause by toggling the stream source
  useEffect(() => {
    if (imgRef.current) {
      if (isPlaying && active && !streamError) {
        imgRef.current.src = streamUrl
      } else {
        imgRef.current.src = "" // Stop fetching stream
      }
    }
  }, [isPlaying, active, streamUrl, streamError])

  const formattedTime = currentTime.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })

  const handleStreamError = () => {
    setStreamError("Failed to load stream. Camera may be unavailable.")
    setIsPlaying(false)
  }

  const togglePlay = (e: React.MouseEvent) => {
    e.stopPropagation()
    setIsPlaying(!isPlaying)
    setStreamError(null) // Reset error when attempting to play
  }

  return (
    <div className="overflow-hidden rounded-lg border group">
      <div className="video-container relative aspect-video bg-black">
        {streamError ? (
          <div className="h-full w-full flex items-center justify-center text-white text-center">
            <p className="text-xs">{streamError}</p>
          </div>
        ) : !isPlaying || !active ? (
          <div className="h-full w-full flex items-center justify-center text-white text-center">
            <p className="text-xs">{!active ? "Camera Offline" : "Stream Paused"}</p>
          </div>
        ) : (
          <img
            ref={imgRef}
            src={isPlaying && active ? streamUrl : ""}
            alt={`${name} camera feed`}
            className="h-full w-full object-cover"
            onError={handleStreamError}
          />
        )}

        {hasAlert && (
          <div className="absolute top-2 right-2 z-10">
            <div className="pulse p-1 rounded-full bg-red-500/20">
              <AlertTriangle className="h-4 w-4 text-red-500" />
            </div>
          </div>
        )}

        <div className="absolute bottom-0 left-0 right-0 flex items-center justify-between p-2 text-white">
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className={`${active ? "bg-green-500/20 text-green-500" : "bg-red-500/20 text-red-500"} border-none`}
            >
              {active ? "LIVE" : "OFFLINE"}
            </Badge>
            <span className="text-xs">{formattedTime}</span>
          </div>
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-white hover:bg-black/50"
              onClick={togglePlay}
            >
              {isPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
            </Button>
            <Button variant="ghost" size="icon" className="h-6 w-6 text-white hover:bg-black/50">
              <Maximize2 className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>
      <div className="p-2">
        <div className="flex items-center justify-between">
          <h3 className="font-medium">{name}</h3>
          {hasAlert && (
            <Badge variant="outline" className="bg-red-500/10 text-red-500 border-red-500/20">
              Alert
            </Badge>
          )}
        </div>
        <p className="text-xs text-muted-foreground">{location}</p>
      </div>
    </div>
  )
}