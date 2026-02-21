"use client"

import { Bell, Moon, Sun, Search, User } from "lucide-react"
import { useTheme } from "next-themes"
import { usePathname } from "next/navigation"
import { useState, useEffect } from "react"

const PAGE_META: Record<string, { title: string; subtitle: string }> = {
  "/": { title: "Dashboard", subtitle: "System Overview & Live Monitoring" },
  "/conversation": { title: "Conversation", subtitle: "AI-Assisted Video Search" },
  "/alerts": { title: "Alerts & Events", subtitle: "Detections and generated alerts" },
  "/analytics": { title: "Analytics", subtitle: "System performance and activity metrics" },
  "/settings": { title: "Settings", subtitle: "Account and system configuration" },
  "/test-dashboard": { title: "Test Dashboard", subtitle: "Run indexers and conversation tests" },
}

export function Header() {
  const { theme, setTheme } = useTheme()
  const pathname = usePathname()
  const [mounted, setMounted] = useState(false)
  useEffect(() => setMounted(true), [])

  const meta = PAGE_META[pathname] || { title: "Terminal AI", subtitle: "" }

  const iconBtn: React.CSSProperties = {
    padding: "7px",
    borderRadius: "var(--radius-md)",
    background: "var(--color-surface-raised)",
    border: "1px solid var(--color-border)",
    color: "var(--color-text-muted)",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    transition: "all 150ms ease",
    boxShadow: "var(--shadow-xs)",
  }

  return (
    <header
      style={{
        height: "58px",
        background: "var(--color-surface-overlay)",
        backdropFilter: "blur(8px)",
        WebkitBackdropFilter: "blur(8px)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-sm)",
        padding: "0 16px",
        display: "flex",
        alignItems: "center",
        gap: "12px",
        flexShrink: 0,
        width: "100%",
        marginBottom: "0",
      }}
    >
      {/* Page title (left) */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <h1 style={{
          fontSize: "0.9375rem",
          fontWeight: 700,
          color: "var(--color-text)",
          lineHeight: 1.2,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}>
          {meta.title}
        </h1>
        {meta.subtitle && (
          <p style={{
            fontSize: "0.6875rem",
            color: "var(--color-text-faint)",
            marginTop: "1px",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
          }}>
            {meta.subtitle}
          </p>
        )}
      </div>

      {/* Search bar (center-ish) */}
      <div style={{ position: "relative", width: "220px", flexShrink: 0 }} className="hidden md:block">
        <Search style={{
          width: "14px", height: "14px",
          position: "absolute", left: "10px", top: "50%", transform: "translateY(-50%)",
          color: "var(--color-text-faint)", pointerEvents: "none",
        }} />
        <input
          type="text"
          placeholder="Search..."
          style={{
            width: "100%",
            background: "var(--color-surface-raised)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            padding: "6px 36px 6px 30px",
            fontSize: "0.8125rem",
            color: "var(--color-text)",
            fontFamily: "var(--font-ui)",
            outline: "none",
            transition: "border-color 150ms",
          }}
          onFocus={e => (e.target.style.borderColor = "var(--color-primary)")}
          onBlur={e => (e.target.style.borderColor = "var(--color-border)")}
        />
        <span style={{
          position: "absolute", right: "8px", top: "50%", transform: "translateY(-50%)",
          fontSize: "0.625rem", fontWeight: 700,
          background: "var(--color-border)",
          borderRadius: "4px",
          padding: "1px 5px",
          color: "var(--color-text-faint)",
          fontFamily: "var(--font-mono)",
          letterSpacing: "0.02em",
        }}>⌘F</span>
      </div>

      {/* Right actions */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <button style={iconBtn} className="header-icon-btn">
          <Bell style={{ width: "15px", height: "15px" }} />
        </button>
        <button
          style={iconBtn}
          className="header-icon-btn"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
        >
          {mounted ? (
            theme === "dark"
              ? <Sun style={{ width: "15px", height: "15px" }} />
              : <Moon style={{ width: "15px", height: "15px" }} />
          ) : (
            <Moon style={{ width: "15px", height: "15px" }} />
          )}
        </button>

        {/* Separator */}
        <div style={{ width: "1px", height: "20px", background: "var(--color-border)", margin: "0 2px" }} />

        {/* User */}
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <div className="hidden md:block" style={{ textAlign: "right" }}>
            <p style={{ fontSize: "0.8125rem", fontWeight: 600, color: "var(--color-text)", lineHeight: 1.2 }}>Admin</p>
            <p style={{ fontSize: "0.6875rem", color: "var(--color-text-faint)" }}>System Admin</p>
          </div>
          <div style={{
            width: "32px", height: "32px",
            borderRadius: "var(--radius-full)",
            background: "linear-gradient(135deg, #15803D 0%, #16A34A 100%)",
            display: "flex", alignItems: "center", justifyContent: "center",
            boxShadow: "var(--shadow-xs)",
            flexShrink: 0,
          }}>
            <User style={{ width: "14px", height: "14px", color: "white" }} />
          </div>
        </div>
      </div>

      <style>{`
        .header-icon-btn:hover {
          background: var(--color-surface) !important;
          color: var(--color-text) !important;
          border-color: var(--color-border-strong) !important;
        }
        .header-icon-btn:active {
          transform: scale(0.95);
        }
      `}</style>
    </header>
  )
}
