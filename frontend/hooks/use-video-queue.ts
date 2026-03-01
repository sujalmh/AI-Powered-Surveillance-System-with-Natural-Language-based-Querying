"use client";

import { useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";

/* ── Types ─────────────────────────────────────────── */

export type IndexingResult = {
  ok: boolean;
  camera_id: number;
  clip_path: string;
  clip_url: string | null;
  indexing: Record<string, any>;
};

export type VideoStatus = "pending" | "uploading" | "indexing" | "completed" | "error";

export type VideoItem = {
  id: string;
  file: File;
  cameraId: number;
  status: VideoStatus;
  result?: IndexingResult;
  error?: string;
  frames?: Array<Record<string, any>>;
};

/* ── Hook ──────────────────────────────────────────── */

export function useVideoQueue() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [globalEverySec, setGlobalEverySec] = useState(1.0);
  const [globalWithCaptions, setGlobalWithCaptions] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* Helpers ------------------------------------------------ */

  const updateVideo = useCallback(
    (id: string, patch: Partial<VideoItem>) =>
      setVideos((prev) => prev.map((v) => (v.id === id ? { ...v, ...patch } : v))),
    [],
  );

  const addFiles = useCallback((files: FileList) => {
    const items: VideoItem[] = Array.from(files).map((file) => ({
      id: crypto.randomUUID(),
      file,
      cameraId: 99,
      status: "pending" as const,
    }));
    setVideos((prev) => [...prev, ...items]);
  }, []);

  const removeVideo = useCallback(
    (id: string) => setVideos((prev) => prev.filter((v) => v.id !== id)),
    [],
  );

  const updateCameraId = useCallback(
    (id: string, cameraId: number) => updateVideo(id, { cameraId }),
    [updateVideo],
  );

  /* Processing --------------------------------------------- */

  const startProcessing = useCallback(async () => {
    setIsProcessing(true);

    const pending = videos.filter((v) => v.status === "pending" || v.status === "error");

    for (const video of pending) {
      updateVideo(video.id, { status: "uploading", error: undefined });

      try {
        const res = await api.uploadVideo(video.file, {
          camera_id: video.cameraId,
          every_sec: globalEverySec,
          with_captions: globalWithCaptions,
        });

        updateVideo(video.id, { status: "indexing", result: res });

        let frames: any[] = [];
        try {
          frames = await api.listClipFrames(res.clip_path, 1000, "yolo");
        } catch {
          /* frame-fetch is best-effort */
        }

        updateVideo(video.id, { status: "completed", result: res, frames: frames || [] });
      } catch (err: any) {
        updateVideo(video.id, { status: "error", error: err?.message || String(err) });
      }
    }

    setIsProcessing(false);
  }, [videos, globalEverySec, globalWithCaptions, updateVideo]);

  /* Derived ------------------------------------------------ */

  const completedCount = videos.filter((v) => v.status === "completed").length;
  const pendingCount = videos.filter((v) => v.status === "pending" || v.status === "error").length;
  const canChat = completedCount > 0;

  return {
    videos,
    globalEverySec,
    setGlobalEverySec,
    globalWithCaptions,
    setGlobalWithCaptions,
    isProcessing,
    fileInputRef,
    addFiles,
    removeVideo,
    updateCameraId,
    startProcessing,
    completedCount,
    pendingCount,
    canChat,
  };
}
