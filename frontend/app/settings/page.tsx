"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { SettingsHeader } from "@/components/settings/settings-header"
import { SettingsTabs } from "@/components/settings/settings-tabs"
import { useState } from "react"

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState("user")

  return (
    <MainLayout>
      <div className="space-y-6">
        <SettingsHeader />
        <SettingsTabs activeTab={activeTab} onTabChange={setActiveTab} />
      </div>
    </MainLayout>
  )
}
