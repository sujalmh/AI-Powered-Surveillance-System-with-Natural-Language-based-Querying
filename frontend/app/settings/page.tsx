"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { SettingsTabs } from "@/components/settings/settings-tabs"
import { useState, useEffect } from "react"
import { useTheme } from "next-themes"
import { Moon, Sun, Monitor } from "lucide-react"
import "./settings.css"

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState("user")
  const [mounted, setMounted] = useState(false)
  const { theme, setTheme } = useTheme()

  // Prevent hydration mismatch by only rendering theme-dependent UI after mount
  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <MainLayout>
      <div className="space-y-6">

        {/* Appearance island */}
        <div className="settings-appearance-card">
          <div>
            <p className="settings-appearance-title">Appearance</p>
            <p className="settings-appearance-subtitle">
              Choose your preferred theme
            </p>
          </div>

          {/* Segmented theme picker */}
          <div className="settings-theme-picker">
            {[
              { value: "light", label: "Light", icon: Sun },
              { value: "system", label: "System", icon: Monitor },
              { value: "dark", label: "Dark", icon: Moon },
            ].map(({ value, label, icon: Icon }) => {
              const isActive = mounted && theme === value
              return (
                <button
                  key={value}
                  type="button"
                  onClick={() => setTheme(value)}
                  className={`theme-button ${isActive ? "theme-button-active" : ""}`}
                >
                  <Icon className="theme-button-icon" />
                  {label}
                </button>
              )
            })}
          </div>
        </div>

        <SettingsTabs activeTab={activeTab} onTabChange={setActiveTab} />
      </div>
    </MainLayout>
  )
}
