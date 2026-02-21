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

export function UserSettings() {
  const [formData, setFormData] = useState({
    fullName: "Admin User",
    email: "admin@surveillance.ai",
    phone: "+1 (555) 123-4567",
    notifications: true,
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
      display: "flex", flexDirection: "column", gap: "16px",
    }}>
      {[
        { label: "Full Name", key: "fullName", type: "text" },
        { label: "Email Address", key: "email", type: "email" },
        { label: "Phone Number", key: "phone", type: "tel" },
      ].map(({ label, key, type }) => (
        <div key={key}>
          <label style={{ fontSize: "0.75rem", fontWeight: 600, color: "var(--color-text-muted)", textTransform: "uppercase" as const, letterSpacing: "0.04em", display: "block", marginBottom: "6px" }}>{label}</label>
          <input
            type={type}
            value={(formData as any)[key]}
            onChange={(e) => setFormData({ ...formData, [key]: e.target.value })}
            style={inputStyle}
            onFocus={e => (e.target.style.borderColor = "var(--color-primary)")}
            onBlur={e => (e.target.style.borderColor = "var(--color-border)")}
          />
        </div>
      ))}

      <label style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer" }}>
        <input
          type="checkbox"
          id="notifications"
          checked={formData.notifications}
          onChange={(e) => setFormData({ ...formData, notifications: e.target.checked })}
          style={{ width: 15, height: 15, accentColor: "var(--color-primary)", cursor: "pointer" }}
        />
        <span style={{ fontSize: "0.875rem", color: "var(--color-text)" }}>Enable email notifications</span>
      </label>

      <div style={{
        display: "flex", justifyContent: "flex-end", gap: "8px",
        paddingTop: "16px", borderTop: "1px solid var(--color-border)", marginTop: "4px",
      }}>
        <button
          style={{
            padding: "8px 16px", borderRadius: "var(--radius-md)",
            border: "1px solid var(--color-border)", background: "transparent",
            color: "var(--color-text-muted)", fontSize: "0.875rem", fontWeight: 500,
            cursor: "pointer", fontFamily: "var(--font-ui)", transition: "all 150ms",
          }}
          onMouseEnter={e => { e.currentTarget.style.background = "var(--color-surface-raised)"; e.currentTarget.style.color = "var(--color-text)"; }}
          onMouseLeave={e => { e.currentTarget.style.background = "transparent"; e.currentTarget.style.color = "var(--color-text-muted)"; }}
        >
          Reset Password
        </button>
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
          {saved ? "Saved!" : "Save Changes"}
        </button>
      </div>
    </div>
  )
}
