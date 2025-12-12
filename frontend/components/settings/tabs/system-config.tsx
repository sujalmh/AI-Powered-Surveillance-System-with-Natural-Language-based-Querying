"use client"

import { useState } from "react"
import { Save, Check, Eye, EyeOff, AlertTriangle, Trash2 } from "lucide-react"

export function SystemConfig() {
  const [formData, setFormData] = useState({
    mongoUri: "mongodb+srv://user:pass@cluster.mongodb.net/surveillance",
    llmApiKey: "sk-proj-...",
    model: "gpt-4-turbo",
    storageLimit: "1000",
  })
  const [indexingMode, setIndexingMode] = useState<"structured" | "semantic" | "both">("both")
  const [modeSaving, setModeSaving] = useState<"idle" | "saving" | "saved">("idle")
  const [showSecrets, setShowSecrets] = useState(false)
  const [saved, setSaved] = useState(false)

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  // Fetch current indexing mode on mount
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useState(() => {
    import("../../../lib/api").then(({ api }) => {
      api.getIndexingMode().then((res) => setIndexingMode(res.indexing_mode)).catch(() => {})
    })
    return undefined
  })

  const handleModeSave = async () => {
    setModeSaving("saving")
    try {
      const { api } = await import("../../../lib/api")
      const res = await api.setIndexingMode(indexingMode)
      setIndexingMode(res.indexing_mode)
      setModeSaving("saved")
      setTimeout(() => setModeSaving("idle"), 1200)
    } catch (e) {
      setModeSaving("idle")
      console.error(e)
      alert("Failed to save indexing mode")
    }
  }

  return (
    <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6 max-w-2xl space-y-6">
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm text-muted-foreground">MongoDB URI</label>
          <button
            onClick={() => setShowSecrets(!showSecrets)}
            className="text-xs text-cyan-400 hover:text-cyan-300 flex items-center gap-1"
          >
            {showSecrets ? <EyeOff className="w-3 h-3" /> : <Eye className="w-3 h-3" />}
            {showSecrets ? "Hide" : "Show"}
          </button>
        </div>
        <input
          type={showSecrets ? "text" : "password"}
          value={formData.mongoUri}
          onChange={(e) => setFormData({ ...formData, mongoUri: e.target.value })}
          className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none font-mono text-xs focus:border-cyan-400/50 transition-colors"
        />
        <p className="text-xs text-muted-foreground mt-1">Database connection string</p>
      </div>

      <div>
        <label className="text-sm text-muted-foreground block mb-2">LLM API Key</label>
        <input
          type={showSecrets ? "text" : "password"}
          value={formData.llmApiKey}
          onChange={(e) => setFormData({ ...formData, llmApiKey: e.target.value })}
          className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none font-mono text-xs focus:border-cyan-400/50 transition-colors"
        />
        <p className="text-xs text-muted-foreground mt-1">OpenAI or compatible API key</p>
      </div>

      <div>
        <label className="text-sm text-muted-foreground block mb-2">Model Selection</label>
        <select
          value={formData.model}
          onChange={(e) => setFormData({ ...formData, model: e.target.value })}
          className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none focus:border-cyan-400/50 transition-colors"
        >
          <option>gpt-4-turbo</option>
          <option>gpt-4</option>
          <option>gpt-3.5-turbo</option>
          <option>claude-3-opus</option>
        </select>
      </div>

      <div>
        <label className="text-sm text-muted-foreground block mb-2">Storage Limit (GB)</label>
        <input
          type="number"
          value={formData.storageLimit}
          onChange={(e) => setFormData({ ...formData, storageLimit: e.target.value })}
          className="w-full bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none font-mono text-xs focus:border-cyan-400/50 transition-colors"
        />
      </div>

      <div>
        <label className="text-sm text-muted-foreground block mb-2">Indexing Mode</label>
        <div className="flex items-center gap-3">
          <select
            value={indexingMode}
            onChange={(e) => setIndexingMode(e.target.value as any)}
            className="flex-1 bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground bg-transparent outline-none focus:border-cyan-400/50 transition-colors"
          >
            <option value="structured">Structured only (YOLO/attributes)</option>
            <option value="semantic">Semantic only (VLM)</option>
            <option value="both">Both (Structured + Semantic)</option>
          </select>
          <button
            onClick={handleModeSave}
            className="glow-button px-4 py-2"
            disabled={modeSaving === "saving"}
            title="Save indexing mode"
          >
            {modeSaving === "saving" ? "Saving..." : modeSaving === "saved" ? "Saved" : "Save"}
          </button>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          Controls whether new indexing/API operations use structured (YOLO), semantic (VLM), or both. Takes effect immediately.
        </p>
      </div>

      <button
        onClick={handleSave}
        className={`glow-button flex items-center gap-2 transition-all ${saved ? "opacity-75" : ""}`}
      >
        {saved ? <Check className="w-4 h-4" /> : <Save className="w-4 h-4" />}
        {saved ? "Saved" : "Save Configuration"}
      </button>

      <div className="pt-8 border-t border-white/10">
        <h3 className="text-red-400 font-medium mb-4 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" /> Danger Zone
        </h3>
        <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
          <h4 className="text-sm font-medium text-red-200 mb-1">Reset System Data</h4>
          <p className="text-xs text-red-300/70 mb-4">
            Permanently delete all videos, clips, snapshots, database records, and search indices.
            This action cannot be undone.
          </p>
          <button
            onClick={async () => {
              if (
                confirm(
                  "Are you ABSOLUTELY SURE?\n\nThis will permanently delete ALL:\n- Videos and recordings\n- Generated clips and snapshots\n- Database records (detections, alerts, metadata)\n- Search indices\n\nThis action cannot be undone."
                )
              ) {
                try {
                  const { api } = await import("../../../lib/api")
                  await api.resetSystem()
                  alert("System reset complete. All data has been wiped.")
                  window.location.reload()
                } catch (e) {
                  console.error(e)
                  alert("Failed to reset system: " + e)
                }
              }
            }}
            className="bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/50 px-4 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-2"
          >
            <Trash2 className="w-3 h-3" />
            Reset System Data
          </button>
        </div>
      </div>
    </div>
  )
}
