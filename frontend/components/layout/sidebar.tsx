"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { BarChart3, MessageSquare, AlertCircle, Settings, Home, Brain, MonitorCheck, LogOut, HelpCircle, User } from "lucide-react"

export function Sidebar() {
  const pathname = usePathname()

  const menuItems = [
    { href: "/", label: "Dashboard", icon: Home },
    { href: "/conversation", label: "Conversation", icon: MessageSquare },
    { href: "/alerts", label: "Alerts", icon: AlertCircle },
    { href: "/analytics", label: "Analytics", icon: BarChart3 },
  ]

  const generalItems = [
    { href: "/settings", label: "Settings", icon: Settings },
    { href: "/test-dashboard", label: "Test Dashboard", icon: MonitorCheck },
  ]

  const NavItem = ({ href, label, icon: Icon }: { href: string; label: string; icon: React.ComponentType<{ className?: string }> }) => {
    const isActive = pathname === href
    return (
      <Link
        href={href}
        style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
          padding: "8px 12px",
          borderRadius: "var(--radius-md)",
          position: "relative",
          textDecoration: "none",
          transition: "background 150ms ease, color 150ms ease",
          background: isActive ? "var(--color-primary-light)" : "transparent",
          color: isActive ? "var(--color-primary)" : "var(--color-text-muted)",
          fontWeight: isActive ? 600 : 500,
          fontSize: "0.875rem",
        }}
        className="group sidebar-nav-item"
        data-active={isActive ? "true" : undefined}
      >
        {/* Left accent bar */}
        {isActive && (
          <span
            style={{
              position: "absolute",
              left: 0,
              top: "6px",
              bottom: "6px",
              width: "3px",
              borderRadius: "0 var(--radius-full) var(--radius-full) 0",
              background: "var(--color-primary)",
            }}
          />
        )}
        <Icon className={`w-4 h-4 shrink-0`} style={{ color: isActive ? "var(--color-primary)" : "var(--color-text-faint)" }} />
        <span>{label}</span>
      </Link>
    )
  }

  return (
    <aside
      style={{
        width: "220px",
        height: "100%",
        background: "var(--color-surface)",
        borderRight: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        boxShadow: "var(--shadow-sm)",
        display: "flex",
        flexDirection: "column",
        flexShrink: 0,
        overflow: "hidden",
      }}
    >
      {/* ── Logo ── */}
      <div style={{
        padding: "18px 16px 14px",
        borderBottom: "1px solid var(--color-border)",
        display: "flex",
        alignItems: "center",
        gap: "10px",
      }}>
        <div style={{
          width: "34px",
          height: "34px",
          borderRadius: "var(--radius-md)",
          background: "linear-gradient(135deg, #15803D 0%, #16A34A 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 0 0 0 var(--color-primary-glow)",
          flexShrink: 0,
          transition: "box-shadow 150ms ease",
        }}
        className="logo-icon"
        >
          <Brain className="w-4 h-4 text-white" />
        </div>
        <div>
          <p style={{ fontWeight: 700, fontSize: "0.9375rem", color: "var(--color-text)", lineHeight: 1.2 }}>
            Terminal AI
          </p>
          <p style={{ fontSize: "0.6875rem", color: "var(--color-text-faint)", letterSpacing: "0.03em", marginTop: "1px" }}>
            Surveillance
          </p>
        </div>
      </div>

      {/* ── Nav ── */}
      <nav style={{ flex: 1, overflowY: "auto", padding: "12px 8px", display: "flex", flexDirection: "column", gap: "2px" }}>

        {/* MENU section */}
        <p style={{
          fontSize: "10px",
          fontWeight: 700,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "var(--color-text-faint)",
          padding: "4px 12px 6px",
          marginTop: "2px",
        }}>
          Menu
        </p>
        {menuItems.map((item) => (
          <NavItem key={item.href} {...item} />
        ))}

        {/* GENERAL section */}
        <p style={{
          fontSize: "10px",
          fontWeight: 700,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
          color: "var(--color-text-faint)",
          padding: "4px 12px 6px",
          marginTop: "12px",
        }}>
          General
        </p>
        {generalItems.map((item) => (
          <NavItem key={item.href} {...item} />
        ))}
        <button
          style={{
            display: "flex", alignItems: "center", gap: "10px",
            padding: "8px 12px", borderRadius: "var(--radius-md)",
            background: "transparent", border: "none",
            color: "var(--color-text-muted)", fontSize: "0.875rem", fontWeight: 500,
            width: "100%", cursor: "pointer",
            transition: "background 150ms, color 150ms",
          }}
          className="sidebar-nav-item"
        >
          <HelpCircle className="w-4 h-4 shrink-0" style={{ color: "var(--color-text-faint)" }} />
          <span>Help</span>
        </button>
      </nav>

      {/* ── Pinned User Area ── */}
      <div style={{
        borderTop: "1px solid var(--color-border)",
        padding: "12px 14px",
        display: "flex",
        alignItems: "center",
        gap: "10px",
        marginTop: "auto",
      }}>
        <div style={{
          width: "30px",
          height: "30px",
          borderRadius: "var(--radius-full)",
          background: "linear-gradient(135deg, #15803D 0%, #16A34A 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}>
          <User className="w-3.5 h-3.5 text-white" />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <p style={{ fontSize: "0.8125rem", fontWeight: 600, color: "var(--color-text)", lineHeight: 1.2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            Admin User
          </p>
          <p style={{ fontSize: "0.6875rem", color: "var(--color-text-faint)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            System Admin
          </p>
        </div>
        <button
          title="Logout"
          style={{
            padding: "4px", borderRadius: "var(--radius-sm)",
            background: "transparent", border: "none",
            color: "var(--color-text-faint)", cursor: "pointer",
            transition: "color 150ms",
          }}
        >
          <LogOut className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Hover styles injected inline via style tag approach */}
      <style>{`
        .sidebar-nav-item:hover:not([data-active]) {
          background: var(--color-surface-raised) !important;
          color: var(--color-text) !important;
        }
        .sidebar-nav-item:hover:not([data-active]) svg {
          color: var(--color-text-muted) !important;
        }
        .logo-icon:hover {
          box-shadow: 0 0 0 4px var(--color-primary-glow) !important;
        }
      `}</style>
    </aside>
  )
}
