"use client";

import React from "react";
import { Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import type { VideoStatus } from "@/hooks/use-video-queue";

const STATUS_MAP: Record<
  VideoStatus,
  { label: string; color: string; bg: string; spin?: boolean; Icon?: React.ElementType }
> = {
  pending:   { label: "Pending",   color: "var(--color-text-muted)",  bg: "var(--color-surface-raised)" },
  uploading: { label: "Uploading", color: "#16A34A",                  bg: "rgba(22,163,74,0.12)", spin: true },
  indexing:  { label: "Indexing",  color: "#7C3AED",                  bg: "rgba(124,58,237,0.10)", spin: true },
  completed: { label: "Done",      color: "#16A34A",                  bg: "rgba(22,163,74,0.12)", Icon: CheckCircle2 },
  error:     { label: "Error",     color: "var(--color-danger)",      bg: "rgba(220,38,38,0.10)", Icon: AlertCircle },
};

export function StatusPill({ status }: { status: VideoStatus }) {
  const s = STATUS_MAP[status];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: "3px 10px",
        borderRadius: "var(--radius-full)",
        fontSize: "0.75rem",
        fontWeight: 600,
        color: s.color,
        background: s.bg,
      }}
    >
      {s.spin && <Loader2 style={{ width: 11, height: 11, animation: "spin 1s linear infinite" }} />}
      {s.Icon && <s.Icon style={{ width: 11, height: 11 }} />}
      {s.label}
    </span>
  );
}
