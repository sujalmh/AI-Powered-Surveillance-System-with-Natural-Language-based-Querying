export function AISteps() {
  const steps = [
    { name: "Parse Query", status: "complete", details: "Extracted: location=Gate 3, time=6 PM, object=people" },
    { name: "Extract Objects", status: "complete", details: "Detected: person (confidence: 0.98)" },
    { name: "MongoDB Query", status: "complete", details: "Found 12 matching events in database" },
    { name: "Vector Search", status: "complete", details: "Searched embeddings for similar scenes" },
    { name: "Fusion & Ranking", status: "in-progress", details: "Combining results by relevance..." },
    { name: "Return Results", status: "pending", details: "Preparing final output" },
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case "complete":
        return "bg-green-500/20 border-green-500/50 text-green-400"
      case "in-progress":
        return "bg-blue-500/20 border-blue-500/50 text-blue-400"
      case "pending":
        return "bg-gray-500/20 border-gray-500/50 text-gray-400"
      default:
        return "bg-gray-500/20 border-gray-500/50 text-gray-400"
    }
  }

  return (
    <div className="w-80 bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6 overflow-y-auto">
      <h3 className="text-lg font-bold text-foreground mb-4">AI Processing Steps</h3>

      <div className="space-y-3">
        {steps.map((step, idx) => (
          <div
            key={idx}
            className={`bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-3 border ${getStatusColor(step.status)}`}
          >
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 mt-1">
                {step.status === "complete" && <div className="w-2 h-2 rounded-full bg-green-400" />}
                {step.status === "in-progress" && <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />}
                {step.status === "pending" && <div className="w-2 h-2 rounded-full bg-gray-400" />}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold">{step.name}</p>
                <p className="text-xs opacity-70 mt-1">{step.details}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="mt-6 bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-4 bg-white/5">
        <p className="text-xs text-muted-foreground mb-2">Response Time</p>
        <p className="text-2xl font-bold text-cyan-400">2.3s</p>
      </div>
    </div>
  )
}
