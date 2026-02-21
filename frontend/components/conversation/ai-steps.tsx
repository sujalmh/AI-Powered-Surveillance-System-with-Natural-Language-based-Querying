export function AISteps() {
  const steps = [
    { name: "Parse Query", status: "complete", details: "location=Gate 3, time=6 PM" },
    { name: "Extract Objects", status: "complete", details: "person (confidence: 0.98)" },
    { name: "MongoDB Query", status: "complete", details: "12 matching events found" },
    { name: "Vector Search", status: "complete", details: "Searched embeddings" },
    { name: "Fusion & Ranking", status: "in-progress", details: "Combining by relevance..." },
    { name: "Return Results", status: "pending", details: "Preparing output" },
  ]

  const getStyle = (status: string) => {
    if (status === "complete") return {
      dot: { background: "var(--color-success)" },
      card: { borderColor: "rgba(22,163,74,0.2)", background: "rgba(22,163,74,0.04)" },
      name: { color: "var(--color-text)" },
    }
    if (status === "in-progress") return {
      dot: { background: "var(--color-primary)", animation: "pulse-dot 1.5s ease-in-out infinite" },
      card: { borderColor: "rgba(74,222,128,0.3)", background: "rgba(22,163,74,0.06)" },
      name: { color: "var(--color-primary)" },
    }
    return {
      dot: { background: "var(--color-border-strong)" },
      card: { borderColor: "var(--color-border)", background: "transparent" },
      name: { color: "var(--color-text-faint)" },
    }
  }

  return (
    <div
      style={{
        width: "240px",
        flexShrink: 0,
        background: "var(--color-surface)",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--radius-lg)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        height: "100%",
      }}
    >
      {/* Header */}
      <div style={{
        height: "52px",
        borderBottom: "1px solid var(--color-border)",
        display: "flex",
        alignItems: "center",
        padding: "0 14px",
        flexShrink: 0,
      }}>
        <span style={{ fontSize: "0.875rem", fontWeight: 600, color: "var(--color-text)" }}>Processing</span>
      </div>

      {/* Steps */}
      <div style={{ flex: 1, overflowY: "auto", padding: "12px", display: "flex", flexDirection: "column", gap: "6px" }}>
        {steps.map((step, idx) => {
          const s = getStyle(step.status)
          return (
            <div key={idx} style={{
              border: `1px solid ${(s.card as any).borderColor}`,
              background: (s.card as any).background,
              borderRadius: "var(--radius-md)",
              padding: "8px 10px",
              display: "flex",
              alignItems: "flex-start",
              gap: "8px",
            }}>
              <div style={{
                width: "6px", height: "6px",
                borderRadius: "50%",
                marginTop: "5px",
                flexShrink: 0,
                ...s.dot,
              }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{ fontSize: "0.8125rem", fontWeight: 600, margin: 0, ...s.name }}>{step.name}</p>
                <p style={{ fontSize: "0.6875rem", color: "var(--color-text-faint)", marginTop: "2px", fontFamily: "var(--font-mono)" }}>{step.details}</p>
              </div>
            </div>
          )
        })}
      </div>

      {/* Footer */}
      <div style={{
        borderTop: "1px solid var(--color-border)",
        padding: "12px 14px",
        flexShrink: 0,
      }}>
        <p style={{ fontSize: "0.6875rem", color: "var(--color-text-muted)", marginBottom: "4px" }}>Response Time</p>
        <p style={{ fontSize: "1.375rem", fontWeight: 700, color: "var(--color-primary)", fontFamily: "var(--font-mono)" }}>2.3s</p>
      </div>

      <style>{`
        @keyframes pulse-dot { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
      `}</style>
    </div>
  )
}
