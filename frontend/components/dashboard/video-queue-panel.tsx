"use client";

import React from "react";
import { Loader2, Video } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { VideoItemRow } from "./video-item-row";
import type { VideoItem } from "@/hooks/use-video-queue";

type Props = {
  videos: VideoItem[];
  globalEverySec: number;
  setGlobalEverySec: (v: number) => void;
  globalWithCaptions: boolean;
  setGlobalWithCaptions: (v: boolean) => void;
  isProcessing: boolean;
  pendingCount: number;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  onFilesSelected: (files: FileList) => void;
  onRemove: (id: string) => void;
  onCameraIdChange: (id: string, cameraId: number) => void;
  onStartProcessing: () => void;
};

export function VideoQueuePanel({
  videos,
  globalEverySec,
  setGlobalEverySec,
  globalWithCaptions,
  setGlobalWithCaptions,
  isProcessing,
  pendingCount,
  fileInputRef,
  onFilesSelected,
  onRemove,
  onCameraIdChange,
  onStartProcessing,
}: Props) {
  const canStart = !isProcessing && pendingCount > 0;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      onFilesSelected(e.target.files);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        gap: "var(--gap-inner)",
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-sm)",
        padding: 16,
        overflow: "hidden",
        minHeight: 0,
      }}
    >
      {/* Settings row */}
      <SettingsBar
        globalEverySec={globalEverySec}
        setGlobalEverySec={setGlobalEverySec}
        globalWithCaptions={globalWithCaptions}
        setGlobalWithCaptions={setGlobalWithCaptions}
        isProcessing={isProcessing}
        fileInputRef={fileInputRef}
        onFileChange={handleFileChange}
      />

      {/* Video list */}
      <ScrollArea style={{ flex: 1, minHeight: 0 }}>
        {videos.length === 0 ? (
          <EmptyState />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "2px 2px" }}>
            {videos.map((v) => (
              <VideoItemRow
                key={v.id}
                video={v}
                isProcessing={isProcessing}
                onCameraIdChange={onCameraIdChange}
                onRemove={onRemove}
              />
            ))}
          </div>
        )}
      </ScrollArea>

      {/* Start button */}
      <div style={{ flexShrink: 0, display: "flex", justifyContent: "flex-end" }}>
        <button
          onClick={onStartProcessing}
          disabled={!canStart}
          style={{
            padding: "9px 22px",
            borderRadius: "var(--radius-md)",
            border: "none",
            background: "var(--color-primary)",
            color: "#fff",
            fontSize: "0.875rem",
            fontWeight: 600,
            cursor: canStart ? "pointer" : "default",
            display: "flex",
            alignItems: "center",
            gap: 8,
            opacity: canStart ? 1 : 0.5,
            transition: "opacity 150ms",
            fontFamily: "var(--font-ui)",
          }}
        >
          {isProcessing && (
            <Loader2 style={{ width: 14, height: 14, animation: "spin 1s linear infinite" }} />
          )}
          {isProcessing ? "Processing Queue…" : "Start Indexing Queue"}
        </button>
      </div>
    </div>
  );
}

/* ── Sub-components ──────────────────────────────── */

function SettingsBar({
  globalEverySec,
  setGlobalEverySec,
  globalWithCaptions,
  setGlobalWithCaptions,
  isProcessing,
  fileInputRef,
  onFileChange,
}: {
  globalEverySec: number;
  setGlobalEverySec: (v: number) => void;
  globalWithCaptions: boolean;
  setGlobalWithCaptions: (v: boolean) => void;
  isProcessing: boolean;
  fileInputRef: React.RefObject<HTMLInputElement | null>;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr auto",
        gap: 12,
        alignItems: "end",
        background: "var(--color-surface-raised)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-md)",
        padding: "12px 14px",
        flexShrink: 0,
      }}
    >
      {/* Sample interval */}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <label
          style={{
            fontSize: "0.75rem",
            fontWeight: 600,
            color: "var(--color-text-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.04em",
          }}
        >
          Sample every (sec)
        </label>
        <input
          type="number"
          step="0.1"
          min={0.1}
          value={globalEverySec}
          onChange={(e) => setGlobalEverySec(parseFloat(e.target.value || "1.0"))}
          style={{
            padding: "7px 10px",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            background: "var(--color-surface)",
            color: "var(--color-text)",
            fontSize: "0.875rem",
            fontFamily: "var(--font-ui)",
            outline: "none",
            transition: "border-color 150ms",
          }}
          onFocus={(e) => (e.target.style.borderColor = "var(--color-primary)")}
          onBlur={(e) => (e.target.style.borderColor = "var(--color-border)")}
        />
      </div>

      {/* Caption toggle */}
      <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer", paddingBottom: 8 }}>
        <input
          type="checkbox"
          checked={globalWithCaptions}
          onChange={(e) => setGlobalWithCaptions(e.target.checked)}
          style={{ width: 15, height: 15, accentColor: "var(--color-primary)", cursor: "pointer" }}
        />
        <span style={{ fontSize: "0.875rem", color: "var(--color-text)", fontWeight: 500 }}>
          Generate captions
        </span>
      </label>

      {/* File selector */}
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={isProcessing}
        className="add-videos-btn"
        style={{
          padding: "8px 16px",
          borderRadius: "var(--radius-md)",
          border: "1.5px solid var(--color-border)",
          background: "var(--color-surface)",
          color: "var(--color-text)",
          fontSize: "0.8125rem",
          fontWeight: 600,
          cursor: isProcessing ? "not-allowed" : "pointer",
          opacity: isProcessing ? 0.5 : 1,
          transition: "all 150ms",
          fontFamily: "var(--font-ui)",
        }}
      >
        + Add Videos
      </button>
      <input
        type="file"
        multiple
        accept="video/mp4"
        ref={fileInputRef}
        style={{ display: "none" }}
        onChange={onFileChange}
      />
    </div>
  );
}

function EmptyState() {
  return (
    <div style={{ padding: "48px 16px", textAlign: "center", color: "var(--color-text-faint)" }}>
      <Video style={{ width: 36, height: 36, margin: "0 auto 10px", opacity: 0.3 }} />
      <p style={{ fontSize: "0.875rem" }}>No videos added yet</p>
    </div>
  );
}
