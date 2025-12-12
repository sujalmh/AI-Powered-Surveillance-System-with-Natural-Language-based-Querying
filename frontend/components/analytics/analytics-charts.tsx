"use client"

import { useEffect, useMemo, useState } from "react"
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts"
import { api } from "@/lib/api"

type ActivityPoint = { time: string; events: number }
type ColorSlice = { name: string; value: number; fill?: string }
type PerfPoint = { camera: string; uptime: number }
type MediaSlice = { name: string; value: number }

const COLOR_MAP: Record<string, string> = {
  Red: "#ef4444",
  Green: "#22c55e",
  Blue: "#3b82f6",
  Yellow: "#f59e0b",
  Black: "#374151",
  White: "#f3f4f6",
  Purple: "#a855f7",
  Orange: "#fb923c",
  Pink: "#ec4899",
  Brown: "#92400e",
  Gray: "#9ca3af",
  Cyan: "#06b6d4",
  Magenta: "#d946ef",
  Lime: "#84cc16",
  Navy: "#1f2937",
  Teal: "#14b8a6",
  Violet: "#8b5cf6",
  Maroon: "#7f1d1d",
  Silver: "#c0c0c0",
  Gold: "#f59e0b",
  Coral: "#fb7185",
  Turquoise: "#22d3ee",
  Salmon: "#f87171",
  Indigo: "#6366f1",
}

export function AnalyticsCharts() {
  const [activity, setActivity] = useState<ActivityPoint[]>([])
  const [colors, setColors] = useState<ColorSlice[]>([])
  const [perf, setPerf] = useState<PerfPoint[]>([])
  const [media, setMedia] = useState<MediaSlice[]>([])
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setErr(null)
    try {
      // Build activity from detections in last 6h
      const params = new URLSearchParams()
      params.set("last_minutes", String(360))
      const dets = await api.listDetections(params)
      const bucket: Record<string, number> = {}
      for (const d of dets as any[]) {
        const ts = d.timestamp ? new Date(d.timestamp) : null
        if (!ts) continue
        const hour = ts.getHours().toString().padStart(2, "0") + ":00"
        bucket[hour] = (bucket[hour] || 0) + (Array.isArray(d.objects) ? d.objects.length : 1)
      }
      const hours = Object.keys(bucket).sort()
      setActivity(hours.map((h) => ({ time: h, events: bucket[h] })))

      // Colors distribution
      const colorCounts = await api.colorCounts({ last_minutes: 360, top_n: 8 })
      setColors(
        (colorCounts as any[]).map((c) => ({
          name: c.color || "Unknown",
          value: c.count || 0,
          fill: COLOR_MAP[c.color] || "#06b6d4",
        }))
      )

      // Camera performance: 100 for running, 0 for stopped
      const cams = await api.listCameras()
      setPerf(
        (cams as any[]).map((c) => ({
          camera: `#${c.camera_id}${c.location ? ` · ${c.location}` : ""}`,
          uptime: c.running ? 100 : 0,
        }))
      )

      // Media usage: show available media counts from API
      const [recs, clips] = await Promise.all([api.listRecordings(), api.listClips()])
      setMedia([
        { name: "Recordings", value: Array.isArray(recs) ? recs.length : 0 },
        { name: "Clips", value: Array.isArray(clips) ? clips.length : 0 },
      ])
    } catch (e: any) {
      setErr(e?.message || "Failed to load analytics")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const activityData = activity
  const alertData = colors
  const performanceData = perf
  const storageData = media

  return (
    <div className="grid grid-cols-2 gap-6">
      <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-lg font-semibold text-foreground">Activity Over Time (last 6h)</h3>
          {loading && <span className="text-xs text-muted-foreground">Loading…</span>}
          {err && <span className="text-xs text-red-400">{err}</span>}
        </div>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={activityData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="time" stroke="rgba(255,255,255,0.5)" />
            <YAxis stroke="rgba(255,255,255,0.5)" />
            <Tooltip contentStyle={{ backgroundColor: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)" }} />
            <Line type="monotone" dataKey="events" stroke="#06b6d4" strokeWidth={2} dot={{ fill: "#06b6d4", r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Color Distribution (last 6h)</h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={alertData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${value}`}
              outerRadius={80}
              dataKey="value"
            >
              {alertData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill || "#06b6d4"} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Camera Running Status</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={performanceData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="camera" stroke="rgba(255,255,255,0.5)" angle={-45} textAnchor="end" height={80} />
            <YAxis stroke="rgba(255,255,255,0.5)" domain={[0, 100]} />
            <Tooltip contentStyle={{ backgroundColor: "rgba(0,0,0,0.8)", border: "1px solid rgba(255,255,255,0.2)" }} />
            <Bar dataKey="uptime" fill="#10b981" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white/10 backdrop-blur-xl border border-white/20 rounded-2xl p-6">
        <h3 className="text-lg font-semibold text-foreground mb-4">Media Items</h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={storageData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${value}`}
              outerRadius={80}
              dataKey="value"
            >
              <Cell fill="#06b6d4" />
              <Cell fill="rgba(255,255,255,0.3)" />
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
