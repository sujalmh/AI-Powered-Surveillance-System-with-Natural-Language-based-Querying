"use client"

import { useState } from "react"
import { Save, Check } from "lucide-react"

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "8px 11px",
  border: "1px solid var(--color-border)",
  borderRadius: "var(--radius-md)",
  background: "var(--color-surface-raised)",
  color: "var(--color-text)",
  fontSize: "0.875rem",
  fontFamily: "var(--font-ui)",
  outline: "none",
  transition: "border-color 150ms",
}

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
    <div style={{
      background: "var(--color-surface)", border: "1px solid var(--color-border)",
      borderRadius: "var(--radius-lg)", padding: "20px 24px",
      boxShadow: "var(--shadow-sm)", maxWidth: 640,
      display: "flex", flexDirection: "column", gap: "20px",
    }}>
      {/* Face blurring toggle */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-text)" }}>Enable Face Blurring</p>
          <p style={{ fontSize: "0.8125rem", color: "var(--color-text-muted)", marginTop: 2 }}>Automatically blur faces in recordings</p>
        </div>
        <input
          type="checkbox"
          checked={settings.faceBlurring}
          onChange={(e) => setSettings({ ...settings, faceBlurring: e.target.checked })}
          style={{ width: 16, height: 16, accentColor: "var(--color-primary)", cursor: "pointer" }}
        />
      </div>

      <div style={{ borderTop: "1px solid var(--color-border)", paddingTop: "20px" }}>
        <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase" as const, letterSpacing: "0.04em", display: "block", marginBottom: "6px" }}>
          Data Retention Period (Days)
        </label>
        <input
          type="number"
          value={settings.retentionDays}
          onChange={(e) => setSettings({ ...settings, retentionDays: Number.parseInt(e.target.value) })}
          style={inputStyle}
          onFocus={e => (e.target.style.borderColor = "var(--color-primary)")}
          onBlur={e => (e.target.style.borderColor = "var(--color-border)")}
        />
        <p style={{ fontSize: "0.75rem", color: "var(--color-text-faint)", marginTop: 4 }}>Automatically delete recordings older than this period</p>
      </div>

      <div style={{ borderTop: "1px solid var(--color-border)", paddingTop: "20px" }}>
        <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase" as const, letterSpacing: "0.04em", display: "block", marginBottom: "6px" }}>
          Role Permissions
        </label>
        <select
          value={settings.rolePermissions}
          onChange={(e) => setSettings({ ...settings, rolePermissions: e.target.value })}
          style={{ ...inputStyle }}
          onFocus={e => (e.target.style.borderColor = "var(--color-primary)")}
          onBlur={e => (e.target.style.borderColor = "var(--color-border)")}
        >
          <option value="admin">Administrator</option>
          <option value="operator">Operator</option>
          <option value="viewer">Viewer</option>
        </select>
      </div>

      <div style={{
        display: "flex", justifyContent: "flex-end",
        paddingTop: "16px", borderTop: "1px solid var(--color-border)",
      }}>
        <button
          onClick={handleSave}
          style={{
            display: "flex", alignItems: "center", gap: "6px",
            padding: "8px 18px", borderRadius: "var(--radius-md)", border: "none",
            background: saved ? "#16A34A" : "var(--color-primary)",
            color: "#fff", fontSize: "0.875rem", fontWeight: 600,
            cursor: "pointer", transition: "all 150ms", fontFamily: "var(--font-ui)",
          }}
        >
          {saved ? <Check style={{ width: 14, height: 14 }} /> : <Save style={{ width: 14, height: 14 }} />}
          {saved ? "Saved!" : "Save Privacy Settings"}
        </button>
      </div>
    </div>
  )
}
