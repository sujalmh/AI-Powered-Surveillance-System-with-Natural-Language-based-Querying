"use client"

import type React from "react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

type VideoModalProps = {
  isOpen: boolean
  onClose: () => void
  video: {
    id: string
    title: string
    timestamp: string
    location: string
    url?: string
  }
}

export function VideoModal({ isOpen, onClose, video }: VideoModalProps) {
  const streamUrl = video.url
  console.log("Video URL:", streamUrl)

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>{video.title}</DialogTitle>
          <DialogDescription>
            {video.location} • {video.timestamp}
          </DialogDescription>
        </DialogHeader>
        <div className="mt-4">
          <video
            src={streamUrl}
            controls
            autoPlay
            className="w-full h-auto rounded-md"
            onError={(e) => {
              console.error("Error loading stream:", e)
              console.error("Failed to load stream")
            }}
          />
        </div>
        <div className="mt-4 flex justify-end">
          <Button onClick={onClose}>Close</Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}