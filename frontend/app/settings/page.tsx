"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { SettingsTabs } from "@/components/settings/settings-tabs"
import { useState } from "react"
import { useTheme } from "next-themes"
import { Moon, Sun, Monitor } from "lucide-react"

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState("user")
  const { theme, setTheme } = useTheme()

  return (
    <MainLayout>
      <div className="space-y-6">

        {/* Appearance island */}
        <div style={{
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-lg)",
          padding: "16px 20px",
          boxShadow: "var(--shadow-sm)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "16px",
        }}>
          <div>
            <p style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--color-text)" }}>Appearance</p>
            <p style={{ fontSize: "0.8125rem", color: "var(--color-text-muted)", marginTop: "2px" }}>
              Choose your preferred theme
            </p>
          </div>

          {/* Segmented theme picker */}
          <div style={{
            display: "flex",
            alignItems: "center",
            background: "var(--color-surface-raised)",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius-md)",
            padding: "3px",
            gap: "2px",
          }}>
            {[
              { value: "light", label: "Light", icon: Sun },
              { value: "system", label: "System", icon: Monitor },
              { value: "dark", label: "Dark", icon: Moon },
            ].map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                onClick={() => setTheme(value)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "5px",
                  padding: "6px 12px",
                  borderRadius: "var(--radius-sm)",
                  border: "none",
                  fontSize: "0.8125rem",
                  fontWeight: 500,
                  cursor: "pointer",
                  transition: "all 150ms ease",
                  background: theme === value ? "var(--color-surface)" : "transparent",
                  color: theme === value ? "var(--color-text)" : "var(--color-text-muted)",
                  boxShadow: theme === value ? "var(--shadow-xs)" : "none",
                  fontFamily: "var(--font-ui)",
                }}
              >
                <Icon style={{ width: "13px", height: "13px", color: theme === value ? "var(--color-primary)" : "inherit" }} />
                {label}
              </button>
            ))}
          </div>
        </div>

        <SettingsTabs activeTab={activeTab} onTabChange={setActiveTab} />
      </div>
    </MainLayout>
  )
}
