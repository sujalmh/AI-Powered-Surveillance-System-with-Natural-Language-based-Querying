export function DashboardHeader() {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-2xl font-bold text-foreground mb-2">Dashboard</h1>
        <p className="text-sm text-muted-foreground">System Overview & Live Monitoring</p>
      </div>
      <div className="glass-card glass-noise rounded-2xl px-4 py-3 mt-2 md:mt-3">
        <p className="text-sm text-muted-foreground mb-2">System Status</p>
        <div
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border border-transparent"
          style={{ backgroundColor: "color-mix(in oklab, var(--success) 15%, transparent)", color: "var(--success)" }}
        >
          <span className="w-2 h-2 rounded-full bg-current" />
          <span className="text-sm font-semibold">Operational</span>
        </div>
      </div>
    </div>
  )
}
