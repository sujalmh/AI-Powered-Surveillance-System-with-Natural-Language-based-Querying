"use client";

import React from "react";
import { FileVideo } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { API_BASE } from "@/lib/api";
import type { VideoItem } from "@/hooks/use-video-queue";

type Props = {
  videos: VideoItem[];
};

export function IndexingResultsPanel({ videos }: Props) {
  const completed = videos.filter((v) => v.status === "completed");

  if (completed.length === 0) {
    return <EmptyResults />;
  }

  return (
    <ScrollArea style={{ flex: 1, minHeight: 0 }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 12, paddingBottom: 24 }}>
        {completed.map((video) => (
          <ResultCard key={video.id} video={video} />
        ))}
      </div>
    </ScrollArea>
  );
}

/* ── ResultCard ──────────────────────────────────── */

function ResultCard({ video }: { video: VideoItem }) {
  const clipUrl = video.result?.clip_url
    ? API_BASE.replace(/\/+$/, "") + video.result.clip_url
    : null;

  const stats = [
    {
      label: "Clip Path",
      value: (
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem", wordBreak: "break-all" as const }}>
          {video.result?.clip_path}
        </span>
      ),
    },
    {
      label: "Clip URL",
      value: clipUrl ? (
        <a
          href={clipUrl}
          target="_blank"
          rel="noreferrer"
          style={{ color: "var(--color-primary)", fontSize: "0.8125rem", textDecoration: "none", fontWeight: 500 }}
        >
          Open Video
        </a>
      ) : (
        "N/A"
      ),
    },
    {
      label: "Frames",
      value: <span style={{ fontWeight: 600, color: "var(--color-text)" }}>{video.frames?.length || 0}</span>,
    },
    {
      label: "Raw Data",
      value: (
        <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.8125rem" }}>
          {Object.keys(video.result?.indexing || {}).length} keys
        </span>
      ),
    },
  ];

  return (
    <div
      style={{
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-sm)",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid var(--color-border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-text)", wordBreak: "break-all" }}>
            {video.file.name}
          </p>
          <p style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: 2 }}>
            Camera ID: {video.cameraId}
          </p>
        </div>
        <span
          style={{
            padding: "3px 10px",
            borderRadius: "var(--radius-full)",
            fontSize: "0.75rem",
            fontWeight: 600,
            color: "var(--color-success)",
            background: "rgba(22,163,74,0.10)",
          }}
        >
          Indexed
        </span>
      </div>

      {/* Body */}
      <div style={{ padding: "14px 16px", display: "flex", flexDirection: "column", gap: 12 }}>
        {/* Stats grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(4, 1fr)",
            gap: 12,
            background: "var(--color-surface-raised)",
            borderRadius: "var(--radius-md)",
            padding: 12,
          }}
        >
          {stats.map(({ label, value }) => (
            <div key={label}>
              <p
                style={{
                  fontSize: "0.6875rem",
                  color: "var(--color-text-faint)",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  marginBottom: 4,
                }}
              >
                {label}
              </p>
              <div style={{ fontSize: "0.8125rem", color: "var(--color-text-muted)" }}>{value}</div>
            </div>
          ))}
        </div>

        {/* Frames table */}
        {video.frames && video.frames.length > 0 && <FramesTable frames={video.frames} />}
      </div>
    </div>
  );
}

/* ── FramesTable ─────────────────────────────────── */

const FRAME_COLS = ["Idx", "Time", "Caption", "Objects"] as const;
const MAX_VISIBLE_FRAMES = 50;

function FramesTable({ frames }: { frames: Array<Record<string, any>> }) {
  const visibleFrames = frames.slice(0, MAX_VISIBLE_FRAMES);

  return (
    <div style={{ border: "1px solid var(--color-border)", borderRadius: "var(--radius-md)", overflow: "hidden" }}>
      <div style={{ maxHeight: 300, overflow: "auto" }}>
        <table style={{ width: "100%", fontSize: "0.75rem", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "var(--color-surface-raised)" }}>
              {FRAME_COLS.map((h) => (
                <th
                  key={h}
                  style={{
                    padding: "8px 10px",
                    fontWeight: 600,
                    color: "var(--color-text-muted)",
                    textAlign: "left",
                    position: "sticky",
                    top: 0,
                    background: "var(--color-surface-raised)",
                    borderBottom: "1px solid var(--color-border)",
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleFrames.map((fr, i) => (
              <tr key={i} style={{ borderBottom: "1px solid var(--color-border)" }} className="result-row">
                <td style={{ padding: "7px 10px", color: "var(--color-text-muted)" }}>{fr.frame_index}</td>
                <td style={{ padding: "7px 10px", color: "var(--color-text-muted)" }}>{fr.frame_ts}</td>
                <td
                  style={{
                    padding: "7px 10px",
                    maxWidth: 200,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: "var(--color-text)",
                  }}
                  title={fr.caption}
                >
                  {fr.caption || "-"}
                </td>
                <td
                  style={{
                    padding: "7px 10px",
                    maxWidth: 150,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    color: "var(--color-text-muted)",
                  }}
                >
                  {Array.isArray(fr.object_captions) ? fr.object_captions.join(", ") : "-"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {frames.length > MAX_VISIBLE_FRAMES && (
        <div
          style={{
            padding: 8,
            textAlign: "center",
            fontSize: "0.75rem",
            color: "var(--color-text-faint)",
            background: "var(--color-surface-raised)",
            borderTop: "1px solid var(--color-border)",
          }}
        >
          Showing first {MAX_VISIBLE_FRAMES} of {frames.length} frames
        </div>
      )}
    </div>
  );
}

/* ── EmptyResults ────────────────────────────────── */

function EmptyResults() {
  return (
    <div style={{ padding: "64px 16px", textAlign: "center", color: "var(--color-text-faint)" }}>
      <FileVideo style={{ width: 40, height: 40, margin: "0 auto 12px", opacity: 0.2 }} />
      <p style={{ fontSize: "0.875rem" }}>No indexing results yet.</p>
      <p style={{ fontSize: "0.8125rem", marginTop: 4, color: "var(--color-text-faint)" }}>
        Process some videos to see results here.
      </p>
    </div>
  );
}
