"use client";

import React from "react";
import { FileVideo, Trash2 } from "lucide-react";
import { StatusPill } from "./status-pill";
import type { VideoItem } from "@/hooks/use-video-queue";

type Props = {
  video: VideoItem;
  isProcessing: boolean;
  onCameraIdChange: (id: string, cameraId: number) => void;
  onRemove: (id: string) => void;
};

export function VideoItemRow({ video, isProcessing, onCameraIdChange, onRemove }: Props) {
  const editable = video.status === "pending" || video.status === "error";

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "10px 12px",
        borderRadius: "var(--radius-md)",
        border: "1px solid var(--color-border)",
        background: "var(--color-surface-raised)",
      }}
    >
      {/* Thumbnail placeholder */}
      <div
        style={{
          width: 36,
          height: 36,
          borderRadius: "var(--radius-sm)",
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        <FileVideo style={{ width: 16, height: 16, color: "var(--color-text-faint)" }} />
      </div>

      {/* File info */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <p
          style={{
            fontSize: "0.8125rem",
            fontWeight: 600,
            color: "var(--color-text)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
          title={video.file.name}
        >
          {video.file.name}
        </p>
        <p style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: 1 }}>
          {formatFileSize(video.file.size)}
        </p>
      </div>

      {/* Controls */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexShrink: 0 }}>
        {/* Camera ID */}
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <label
            style={{
              fontSize: "0.625rem",
              fontWeight: 700,
              color: "var(--color-text-faint)",
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            Camera ID
          </label>
          <input
            type="number"
            min={0}
            value={video.cameraId}
            onChange={(e) => onCameraIdChange(video.id, parseInt(e.target.value || "0", 10))}
            disabled={!editable}
            style={{
              width: 72,
              padding: "5px 8px",
              fontSize: "0.8125rem",
              border: "1px solid var(--color-border)",
              borderRadius: "var(--radius-sm)",
              background: "var(--color-surface)",
              color: "var(--color-text)",
              fontFamily: "var(--font-ui)",
              outline: "none",
              opacity: editable ? 1 : 0.5,
            }}
          />
        </div>

        {/* Status */}
        <div style={{ width: 90, display: "flex", justifyContent: "center" }}>
          <StatusPill status={video.status} />
        </div>

        {/* Delete */}
        <button
          onClick={() => onRemove(video.id)}
          disabled={isProcessing && !editable}
          className="delete-video-btn"
          style={{
            width: 30,
            height: 30,
            borderRadius: "var(--radius-sm)",
            border: "none",
            background: "transparent",
            color: "var(--color-text-faint)",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            transition: "all 150ms",
          }}
        >
          <Trash2 style={{ width: 14, height: 14 }} />
        </button>
      </div>
    </div>
  );
}

/* ── Helpers ─────────────────────────────────────── */

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
