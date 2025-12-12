"use client"

import { useState } from "react"
import { Save, Check } from "lucide-react"

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
    <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6 max-w-2xl space-y-6">
      <div>
        <label className="text-sm text-muted-foreground block mb-2">Full Name</label>
        <input
          type="text"
          value={formData.fullName}
          onChange={(e) => setFormData({ ...formData, fullName: e.target.value })}
          className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none focus:border-cyan-400/50 transition-colors"
        />
      </div>

      <div>
        <label className="text-sm text-muted-foreground block mb-2">Email Address</label>
        <input
          type="email"
          value={formData.email}
          onChange={(e) => setFormData({ ...formData, email: e.target.value })}
          className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none focus:border-cyan-400/50 transition-colors"
        />
      </div>

      <div>
        <label className="text-sm text-muted-foreground block mb-2">Phone Number</label>
        <input
          type="tel"
          value={formData.phone}
          onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
          className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none focus:border-cyan-400/50 transition-colors"
        />
      </div>

      <div className="flex items-center gap-3">
        <input
          type="checkbox"
          id="notifications"
          checked={formData.notifications}
          onChange={(e) => setFormData({ ...formData, notifications: e.target.checked })}
          className="w-4 h-4 rounded"
        />
        <label htmlFor="notifications" className="text-sm text-foreground cursor-pointer">
          Enable email notifications
        </label>
      </div>

      <div className="flex gap-3 pt-4">
        <button
          onClick={handleSave}
          className={`glow-button flex items-center gap-2 transition-all ${saved ? "opacity-75" : ""}`}
        >
          {saved ? <Check className="w-4 h-4" /> : <Save className="w-4 h-4" />}
          {saved ? "Saved" : "Save Changes"}
        </button>
        <button className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-6 py-2 text-foreground transition-all duration-300 hover:bg-white/15 hover:border-white/30 hover:shadow-lg">
          Reset Password
        </button>
      </div>
    </div>
  )
}
