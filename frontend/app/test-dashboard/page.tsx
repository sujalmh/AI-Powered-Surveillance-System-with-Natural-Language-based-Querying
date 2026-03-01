"use client";

import React, { useState, useCallback } from "react";
import { Video, MessageSquare, FileVideo, FlaskConical } from "lucide-react";
import { MainLayout } from "@/components/layout/main-layout";
import { API_BASE } from "@/lib/api";
import { ChatInterface } from "@/components/conversation/chat-interface";
import { VideoQueuePanel } from "@/components/dashboard/video-queue-panel";
import { IndexingResultsPanel } from "@/components/dashboard/indexing-results-panel";
import { useVideoQueue } from "@/hooks/use-video-queue";

/* ── Tab definitions ─────────────────────────────── */

const TABS = [
  { id: "videos", label: "Videos", icon: Video },
  { id: "results", label: "Indexing Results", icon: FileVideo },
  { id: "conversation", label: "Conversation", icon: MessageSquare },
] as const;

type TabId = (typeof TABS)[number]["id"];

/* ── Page ────────────────────────────────────────── */

export default function TestDashboardPage() {
  const queue = useVideoQueue();
  const [activeTab, setActiveTab] = useState<TabId>("videos");

  const handleStartProcessing = useCallback(async () => {
    await queue.startProcessing();
    if (queue.completedCount > 0) setActiveTab("results");
  }, [queue]);

  return (
    <MainLayout>
      <div style={{ display: "flex", flexDirection: "column", gap: "var(--gap-island)", height: "100%" }}>
        {/* Page header */}
        <PageHeader />

        {/* Tab bar */}
        <TabBar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          videoCount={queue.videos.length}
          completedCount={queue.completedCount}
          canChat={queue.canChat}
        />

        {/* Tab panels */}
        {activeTab === "videos" && (
          <VideoQueuePanel
            videos={queue.videos}
            globalEverySec={queue.globalEverySec}
            setGlobalEverySec={queue.setGlobalEverySec}
            globalWithCaptions={queue.globalWithCaptions}
            setGlobalWithCaptions={queue.setGlobalWithCaptions}
            isProcessing={queue.isProcessing}
            pendingCount={queue.pendingCount}
            fileInputRef={queue.fileInputRef}
            onFilesSelected={queue.addFiles}
            onRemove={queue.removeVideo}
            onCameraIdChange={queue.updateCameraId}
            onStartProcessing={handleStartProcessing}
          />
        )}

        {activeTab === "results" && <IndexingResultsPanel videos={queue.videos} />}

        {activeTab === "conversation" && <ConversationPanel canChat={queue.canChat} />}
      </div>

      <style>{`
        .test-tab-btn:hover:not([data-active]):not(:disabled) {
          background: var(--color-surface-raised) !important;
          color: var(--color-text) !important;
        }
        .add-videos-btn:hover:not(:disabled) {
          background: var(--color-surface-raised) !important;
          border-color: var(--color-border-strong) !important;
        }
        .delete-video-btn:hover:not(:disabled) {
          color: var(--color-danger) !important;
          background: rgba(220,38,38,0.08) !important;
        }
        .result-row:hover { background: var(--color-surface-raised); }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </MainLayout>
  );
}

/* ── PageHeader ──────────────────────────────────── */

function PageHeader() {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: "var(--radius-md)",
            background: "var(--color-primary-light)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <FlaskConical style={{ width: 16, height: 16, color: "var(--color-primary)" }} />
        </div>
        <div>
          <h1 style={{ fontSize: "1rem", fontWeight: 700, color: "var(--color-text)", lineHeight: 1.2 }}>
            Test Dashboard
          </h1>
          <p style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: 1 }}>
            Batch process videos and query results
          </p>
        </div>
      </div>
      <a
        href={`${API_BASE.replace(/\/+$/, "")}/docs`}
        target="_blank"
        rel="noreferrer"
        style={{ fontSize: "0.8125rem", color: "var(--color-primary)", textDecoration: "none", fontWeight: 500 }}
        onMouseEnter={(e) => (e.currentTarget.style.textDecoration = "underline")}
        onMouseLeave={(e) => (e.currentTarget.style.textDecoration = "none")}
      >
        Backend Docs →
      </a>
    </div>
  );
}

/* ── TabBar ──────────────────────────────────────── */

function TabBadge({ count }: { count: number }) {
  if (count <= 0) return null;
  return (
    <span
      style={{
        background: "var(--color-primary)",
        color: "#fff",
        borderRadius: "var(--radius-full)",
        fontSize: "0.6875rem",
        fontWeight: 700,
        padding: "0 6px",
        lineHeight: "18px",
        minWidth: 18,
        textAlign: "center",
      }}
    >
      {count}
    </span>
  );
}

function TabBar({
  activeTab,
  setActiveTab,
  videoCount,
  completedCount,
  canChat,
}: {
  activeTab: TabId;
  setActiveTab: (id: TabId) => void;
  videoCount: number;
  completedCount: number;
  canChat: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        gap: 4,
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        padding: 4,
        boxShadow: "var(--shadow-sm)",
        flexShrink: 0,
      }}
    >
      {TABS.map(({ id, label, icon: Icon }) => {
        const active = activeTab === id;
        const disabled = id === "conversation" && !canChat;
        return (
          <button
            key={id}
            disabled={disabled}
            onClick={() => !disabled && setActiveTab(id)}
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
              padding: "8px 12px",
              borderRadius: "var(--radius-md)",
              border: "none",
              fontSize: "0.8125rem",
              fontWeight: active ? 600 : 500,
              cursor: disabled ? "not-allowed" : "pointer",
              background: active ? "var(--color-primary-light)" : "transparent",
              color: active
                ? "var(--color-primary)"
                : disabled
                  ? "var(--color-text-faint)"
                  : "var(--color-text-muted)",
              transition: "all 150ms ease",
              fontFamily: "var(--font-ui)",
            }}
            className="test-tab-btn"
            data-active={active ? "true" : undefined}
          >
            <Icon style={{ width: 14, height: 14 }} />
            {label}
            {id === "videos" && <TabBadge count={videoCount} />}
            {id === "results" && <TabBadge count={completedCount} />}
          </button>
        );
      })}
    </div>
  );
}

/* ── ConversationPanel ───────────────────────────── */

function ConversationPanel({ canChat }: { canChat: boolean }) {
  return (
    <div
      style={{
        flex: 1,
        minHeight: 0,
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-sm)",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {canChat ? (
        <ChatInterface onShowSteps={() => {}} />
      ) : (
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--color-text-faint)",
          }}
        >
          <MessageSquare style={{ width: 40, height: 40, marginBottom: 12, opacity: 0.2 }} />
          <p style={{ fontSize: "0.875rem" }}>Conversation unavailable</p>
          <p style={{ fontSize: "0.8125rem", marginTop: 4 }}>Index at least one video to start chatting.</p>
        </div>
      )}
    </div>
  );
}
