"use client"

import { UserSettings } from "./tabs/user-settings"
import { SystemConfig } from "./tabs/system-config"
import { CameraSetup } from "./tabs/camera-setup"
import { PrivacySettings } from "./tabs/privacy-settings"

interface SettingsTabsProps {
  activeTab: string
  onTabChange: (tab: string) => void
}

export function SettingsTabs({ activeTab, onTabChange }: SettingsTabsProps) {
  const tabs = [
    { id: "user", label: "User Settings" },
    { id: "system", label: "System Config" },
    { id: "camera", label: "Camera Setup" },
    { id: "privacy", label: "Privacy" },
  ]

  return (
    <div className="space-y-6">
      <div className="flex gap-2 border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`px-4 py-3 font-medium transition-colors border-b-2 ${
              activeTab === tab.id
                ? "border-cyan-400 text-cyan-400"
                : "border-transparent text-muted-foreground hover:text-foreground"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div>
        {activeTab === "user" && <UserSettings />}
        {activeTab === "system" && <SystemConfig />}
        {activeTab === "camera" && <CameraSetup />}
        {activeTab === "privacy" && <PrivacySettings />}
      </div>
    </div>
  )
}
