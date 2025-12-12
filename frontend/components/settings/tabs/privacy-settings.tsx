"use client"

import { useState } from "react"
import { Save, Check } from "lucide-react"

export function PrivacySettings() {
  const [settings, setSettings] = useState({
    faceBlurring: true,
    retentionDays: 30,
    rolePermissions: "admin",
  })
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6 max-w-2xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="font-medium text-foreground">Enable Face Blurring</p>
          <p className="text-sm text-muted-foreground mt-1">Automatically blur faces in recordings</p>
        </div>
        <input
          type="checkbox"
          checked={settings.faceBlurring}
          onChange={(e) => setSettings({ ...settings, faceBlurring: e.target.checked })}
          className="w-5 h-5 rounded"
        />
      </div>

      <div className="border-t border-border pt-6">
        <label className="text-sm text-muted-foreground block mb-2">Data Retention Period (Days)</label>
        <input
          type="number"
          value={settings.retentionDays}
          onChange={(e) => setSettings({ ...settings, retentionDays: Number.parseInt(e.target.value) })}
          className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none focus:border-cyan-400/50 transition-colors"
        />
        <p className="text-xs text-muted-foreground mt-1">Automatically delete recordings older than this period</p>
      </div>

      <div className="border-t border-border pt-6">
        <label className="text-sm text-muted-foreground block mb-2">Role Permissions</label>
        <select
          value={settings.rolePermissions}
          onChange={(e) => setSettings({ ...settings, rolePermissions: e.target.value })}
          className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none focus:border-cyan-400/50 transition-colors"
        >
          <option value="admin">Administrator</option>
          <option value="operator">Operator</option>
          <option value="viewer">Viewer</option>
        </select>
      </div>

      <button
        onClick={handleSave}
        className={`glow-button flex items-center gap-2 transition-all ${saved ? "opacity-75" : ""}`}
      >
        {saved ? <Check className="w-4 h-4" /> : <Save className="w-4 h-4" />}
        {saved ? "Saved" : "Save Privacy Settings"}
      </button>
    </div>
  )
}
