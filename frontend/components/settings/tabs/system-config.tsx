"use client"

import { useState, useEffect } from "react"
import { Save, Check, Eye, EyeOff, AlertTriangle, Trash2 } from "lucide-react"

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

const monoInputStyle: React.CSSProperties = {
  ...inputStyle,
  fontFamily: "var(--font-mono)",
  fontSize: "0.8125rem",
}

const labelStyle: React.CSSProperties = {
  fontSize: "0.75rem", fontWeight: 600,
  color: "var(--color-text-muted)",
  textTransform: "uppercase",
  letterSpacing: "0.04em",
  display: "block", marginBottom: "6px",
}

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
      await api.setLlmConfig({ provider: formData.provider, model: formData.model, api_key: formData.llmApiKey })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      console.error(e)
      alert("Failed to save LLM configuration: " + e)
    }
  }

  useEffect(() => {
    import("../../../lib/api").then(({ api }) => {
      api.getIndexingMode().then((res) => setIndexingMode(res.indexing_mode)).catch(() => { })
      api.getLlmConfig().then((res) => {
        setFormData(prev => ({ ...prev, provider: res.provider || "openai", model: res.model || "gpt-4o-mini", llmApiKey: res.api_key || "" }))
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
      alert("Failed to save indexing mode")
    }
  }

  const PROVIDER_MODELS: Record<string, string[]> = {
    openai: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    openrouter: ["google/gemini-2.0-flash-001", "google/gemini-2.0-pro-exp-02-05:free", "anthropic/claude-3.5-sonnet", "deepseek/deepseek-r1-distill-llama-70b", "meta-llama/llama-3.1-405b", "mistralai/mistral-large-2411"],
    ollama: ["llama3.1", "mistral", "phi3", "nomic-embed-text"],
  }

  const handleProviderChange = (provider: string) => {
    const models = PROVIDER_MODELS[provider] || []
    setFormData({ ...formData, provider, model: models[0] || formData.model })
  }

  const focusBorder = (e: React.FocusEvent<HTMLInputElement | HTMLSelectElement>) => (e.target.style.borderColor = "var(--color-primary)")
  const blurBorder = (e: React.FocusEvent<HTMLInputElement | HTMLSelectElement>) => (e.target.style.borderColor = "var(--color-border)")

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px", maxWidth: 640 }}>
      {/* Main config island */}
      <div style={{
        background: "var(--color-surface)", border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)", padding: "20px 24px",
        boxShadow: "var(--shadow-sm)", display: "flex", flexDirection: "column", gap: "16px",
      }}>
        {/* MongoDB URI */}
        <div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px" }}>
            <label style={labelStyle}>MongoDB URI</label>
            <button
              onClick={() => setShowSecrets(!showSecrets)}
              style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "0.75rem", color: "var(--color-primary)", background: "none", border: "none", cursor: "pointer" }}
            >
              {showSecrets ? <EyeOff style={{ width: 12, height: 12 }} /> : <Eye style={{ width: 12, height: 12 }} />}
              {showSecrets ? "Hide" : "Show"}
            </button>
          </div>
          <input type={showSecrets ? "text" : "password"} value={formData.mongoUri}
            onChange={(e) => setFormData({ ...formData, mongoUri: e.target.value })}
            style={monoInputStyle} onFocus={focusBorder} onBlur={blurBorder} />
          <p style={{ fontSize: "0.75rem", color: "var(--color-text-faint)", marginTop: 4 }}>Database connection string</p>
        </div>

        {/* LLM API Key */}
        <div>
          <label style={labelStyle}>LLM API Key</label>
          <input type={showSecrets ? "text" : "password"} value={formData.llmApiKey}
            onChange={(e) => setFormData({ ...formData, llmApiKey: e.target.value })}
            style={monoInputStyle} onFocus={focusBorder} onBlur={blurBorder} />
          <p style={{ fontSize: "0.75rem", color: "var(--color-text-faint)", marginTop: 4 }}>OpenAI or compatible API key</p>
        </div>

        {/* Provider */}
        <div>
          <label style={labelStyle}>LLM Provider</label>
          <select value={formData.provider || "openai"} onChange={(e) => handleProviderChange(e.target.value)}
            style={{ ...inputStyle, marginBottom: "12px" }} onFocus={focusBorder} onBlur={blurBorder}>
            <option value="openai">OpenAI</option>
            <option value="openrouter">OpenRouter</option>
            <option value="ollama">Ollama (Local)</option>
          </select>
          <label style={labelStyle}>Model Selection</label>
          <select value={formData.model} onChange={(e) => setFormData({ ...formData, model: e.target.value })}
            style={inputStyle} onFocus={focusBorder} onBlur={blurBorder}>
            {(() => {
              const options = PROVIDER_MODELS[formData.provider || "openai"] || []
              const finalOptions = formData.model && !options.includes(formData.model) ? [formData.model, ...options] : options
              return finalOptions.map((m) => <option key={m} value={m}>{m}</option>)
            })()}
          </select>
        </div>

        {/* Storage Limit */}
        <div>
          <label style={labelStyle}>Storage Limit (GB)</label>
          <input type="number" value={formData.storageLimit}
            onChange={(e) => setFormData({ ...formData, storageLimit: e.target.value })}
            style={monoInputStyle} onFocus={focusBorder} onBlur={blurBorder} />
        </div>

        {/* Indexing mode */}
        <div>
          <label style={labelStyle}>Indexing Mode</label>
          <div style={{ display: "flex", gap: "8px" }}>
            <select value={indexingMode} onChange={(e) => setIndexingMode(e.target.value as any)}
              style={{ ...inputStyle, flex: 1 }} onFocus={focusBorder} onBlur={blurBorder}>
              <option value="structured">Structured only (YOLO/attributes)</option>
              <option value="semantic">Semantic only (VLM)</option>
              <option value="both">Both (Structured + Semantic)</option>
            </select>
            <button
              onClick={handleModeSave}
              disabled={modeSaving === "saving"}
              style={{
                padding: "8px 16px", borderRadius: "var(--radius-md)", border: "none",
                background: "var(--color-primary)", color: "#fff",
                fontSize: "0.875rem", fontWeight: 600, cursor: "pointer",
                opacity: modeSaving === "saving" ? 0.6 : 1, transition: "opacity 150ms",
                fontFamily: "var(--font-ui)", flexShrink: 0,
              }}
            >
              {modeSaving === "saving" ? "Saving…" : modeSaving === "saved" ? "Saved ✓" : "Save"}
            </button>
          </div>
          <p style={{ fontSize: "0.75rem", color: "var(--color-text-faint)", marginTop: 4 }}>Controls whether new indexing uses structured (YOLO), semantic (VLM), or both. Takes effect immediately.</p>
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", paddingTop: "16px", borderTop: "1px solid var(--color-border)" }}>
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
            {saved ? "Saved!" : "Save Configuration"}
          </button>
        </div>
      </div>

      {/* Danger zone */}
      <div style={{
        background: "rgba(220,38,38,0.04)", border: "1px solid rgba(220,38,38,0.18)",
        borderRadius: "var(--radius-lg)", padding: "20px 24px",
      }}>
        <h3 style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-danger)", display: "flex", alignItems: "center", gap: "6px", marginBottom: "8px" }}>
          <AlertTriangle style={{ width: 14, height: 14 }} /> Danger Zone
        </h3>
        <p style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-text)", marginBottom: 4 }}>Reset System Data</p>
        <p style={{ fontSize: "0.8125rem", color: "var(--color-text-muted)", marginBottom: "16px" }}>
          Permanently delete all videos, clips, snapshots, database records, and search indices. This action cannot be undone.
        </p>
        <button
          onClick={async () => {
            if (confirm("Are you ABSOLUTELY SURE?\n\nThis will permanently delete ALL:\n- Videos and recordings\n- Generated clips and snapshots\n- Database records (detections, alerts, metadata)\n- Search indices\n\nThis action cannot be undone.")) {
              try {
                const { api } = await import("../../../lib/api")
                await api.resetSystem()
                alert("System reset complete. All data has been wiped.")
                window.location.reload()
              } catch (e) {
                alert("Failed to reset system: " + e)
              }
            }
          }}
          style={{
            display: "flex", alignItems: "center", gap: "6px",
            padding: "8px 16px", borderRadius: "var(--radius-md)",
            border: "1px solid rgba(220,38,38,0.35)",
            background: "rgba(220,38,38,0.08)", color: "var(--color-danger)",
            fontSize: "0.8125rem", fontWeight: 600, cursor: "pointer",
            transition: "all 150ms", fontFamily: "var(--font-ui)",
          }}
          onMouseEnter={e => { e.currentTarget.style.background = "rgba(220,38,38,0.15)"; }}
          onMouseLeave={e => { e.currentTarget.style.background = "rgba(220,38,38,0.08)"; }}
        >
          <Trash2 style={{ width: 13, height: 13 }} />
          Reset System Data
        </button>
      </div>
    </div>
  )
}
