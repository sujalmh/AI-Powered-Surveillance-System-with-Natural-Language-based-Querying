"use client"

import { useState, useEffect } from "react"
import { Save, Check, Eye, EyeOff, AlertTriangle, Trash2 } from "lucide-react"

export function SystemConfig() {
  const [formData, setFormData] = useState({
    mongoUri: "mongodb+srv://user:pass@cluster.mongodb.net/surveillance",
    llmApiKey: "sk-proj-...",
    provider: "openai",
    model: "gpt-4o-mini",
    storageLimit: "1000",
  })
  const [indexingMode, setIndexingMode] = useState<"structured" | "semantic" | "both">("both")
  const [modeSaving, setModeSaving] = useState<"idle" | "saving" | "saved">("idle")
  const [showSecrets, setShowSecrets] = useState(false)
  const [saved, setSaved] = useState(false)

  const handleSave = async () => {
    try {
      const { api } = await import("../../../lib/api")
      await api.setLlmConfig({
        provider: formData.provider,
        model: formData.model,
        api_key: formData.llmApiKey
      })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      console.error(e)
      alert("Failed to save LLM configuration: " + e)
    }
  }

  // Fetch current indexing mode and LLM config on mount
  useEffect(() => {
    import("../../../lib/api").then(({ api }) => {
      api.getIndexingMode().then((res) => setIndexingMode(res.indexing_mode)).catch(() => { })
      api.getLlmConfig().then((res) => {
        setFormData(prev => ({
          ...prev,
          provider: res.provider || "openai",
          model: res.model || "gpt-4o-mini",
          llmApiKey: res.api_key || ""
        }))
      }).catch(() => { })
    })
  }, [])

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

  const PROVIDER_MODELS: Record<string, string[]> = {
    openai: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    openrouter: [
      "google/gemini-2.0-flash-001",
      "google/gemini-2.0-pro-exp-02-05:free",
      "anthropic/claude-3.5-sonnet",
      "deepseek/deepseek-r1-distill-llama-70b",
      "meta-llama/llama-3.1-405b",
      "mistralai/mistral-large-2411",
    ],
    ollama: ["llama3.1", "mistral", "phi3", "nomic-embed-text"],
    anthropic: ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
  }

  const handleProviderChange = (provider: string) => {
    const models = PROVIDER_MODELS[provider] || []
    setFormData({
      ...formData,
      provider,
      model: models[0] || formData.model
    })
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
        <label className="text-sm text-muted-foreground block mb-2">LLM Provider</label>
        <select
          value={formData.provider || "openai"}
          onChange={(e) => handleProviderChange(e.target.value)}
          className="w-full bg-slate-900/50 dark:bg-slate-900 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground outline-none focus:border-cyan-400/50 transition-colors mb-4 [&>option]:bg-slate-900 [&>option]:text-white"
        >
          <option value="openai">OpenAI</option>
          <option value="openrouter">OpenRouter</option>
          <option value="ollama">Ollama (Local)</option>
          <option value="anthropic">Anthropic</option>
        </select>

        <label className="text-sm text-muted-foreground block mb-2">Model Selection</label>
        <select
          value={formData.model}
          onChange={(e) => setFormData({ ...formData, model: e.target.value })}
          className="w-full bg-slate-900/50 dark:bg-slate-900 backdrop-blur-xl border border-white/20 rounded-2xl px-4 py-2 text-foreground outline-none focus:border-cyan-400/50 transition-colors [&>option]:bg-slate-900 [&>option]:text-white"
        >
          {(PROVIDER_MODELS[formData.provider || "openai"] || []).map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
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
