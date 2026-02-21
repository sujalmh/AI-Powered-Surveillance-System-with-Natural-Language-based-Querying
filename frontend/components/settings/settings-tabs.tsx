"use client"

import { UserSettings } from "./tabs/user-settings"
import { SystemConfig } from "./tabs/system-config"
import { CameraSetup } from "./tabs/camera-setup"
import { PrivacySettings } from "./tabs/privacy-settings"

interface SettingsTabsProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

const TABS = [
  { id: "user",    label: "User Settings" },
  { id: "system",  label: "System Config" },
  { id: "camera",  label: "Camera Setup" },
  { id: "privacy", label: "Privacy" },
]

export function SettingsTabs({ activeTab, onTabChange }: SettingsTabsProps) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {/* Tab bar */}
      <div style={{
        display: "flex", gap: "2px",
        background: "var(--color-surface)", border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)", padding: "4px",
        boxShadow: "var(--shadow-sm)",
      }}>
        {TABS.map((tab) => {
          const active = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              style={{
                flex: 1, padding: "8px 14px",
                borderRadius: "var(--radius-md)", border: "none",
                fontSize: "0.8125rem", fontWeight: active ? 600 : 500,
                cursor: "pointer", transition: "all 150ms",
                background: active ? "var(--color-primary-light)" : "transparent",
                color: active ? "var(--color-primary)" : "var(--color-text-muted)",
                fontFamily: "var(--font-ui)",
              }}
              className="settings-tab-btn"
              data-active={active ? "true" : undefined}
            >
              {tab.label}
            </button>
          )
        })}
      </div>

      {/* Tab content */}
      <div>
        {activeTab === "user"    && <UserSettings />}
        {activeTab === "system"  && <SystemConfig />}
        {activeTab === "camera"  && <CameraSetup />}
        {activeTab === "privacy" && <PrivacySettings />}
      </div>

      <style>{`
        .settings-tab-btn:hover:not([data-active]) {
          background: var(--color-surface-raised) !important;
          color: var(--color-text) !important;
        }
      `}</style>
    </div>
  )
}
