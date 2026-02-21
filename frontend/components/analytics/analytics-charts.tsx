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
import { BarChart3, PieChart as PieIcon, Activity, Database } from "lucide-react"

type ActivityPoint = { time: string; events: number }
type ColorSlice = { name: string; value: number; fill?: string }
type PerfPoint = { camera: string; uptime: number }
type MediaSlice = { name: string; value: number }

const COLOR_MAP: Record<string, string> = {
  Red: "#ef4444", Green: "#22c55e", Blue: "#3b82f6", Yellow: "#f59e0b",
  Black: "#374151", White: "#f3f4f6", Purple: "#a855f7", Orange: "#fb923c",
  Pink: "#ec4899", Brown: "#92400e", Gray: "#9ca3af", Cyan: "#06b6d4",
  Magenta: "#d946ef", Lime: "#84cc16", Navy: "#1f2937", Teal: "#14b8a6",
  Violet: "#8b5cf6", Maroon: "#7f1d1d", Silver: "#c0c0c0", Gold: "#f59e0b",
  Coral: "#fb7185", Turquoise: "#22d3ee", Salmon: "#f87171", Indigo: "#6366f1",
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

      const colorCounts = await api.colorCounts({ last_minutes: 360, top_n: 8 })
      setColors(
        (colorCounts as any[]).map((c) => ({
          name: c.color || "Unknown",
          value: c.count || 0,
          fill: COLOR_MAP[c.color] || "#06b6d4",
        }))
      )

      const cams = await api.listCameras()
      setPerf(
        (cams as any[]).map((c) => ({
          camera: `#${c.camera_id}${c.location ? ` · ${c.location}` : ""}`,
          uptime: c.running ? 100 : 0,
        }))
      )

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

  useEffect(() => { load() }, [])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)', padding: '16px', boxShadow: 'var(--shadow-sm)', display: 'flex', flexDirection: 'column' }}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-base font-medium text-stone-900 dark:text-stone-200 tracking-tight">Activity Over Time (last 6h)</h3>
          {loading && <span className="text-sm" style={{ color: 'var(--color-text-muted)' }}>Loading...</span>}
          {err && <span className="text-xs text-rose-500">{err}</span>}
        </div>
        {!loading && activity.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-stone-500 dark:text-stone-400 gap-2 min-h-[250px]">
            <Activity className="w-8 h-8 opacity-50" />
            <span className="text-sm">No data available for selected time range</span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={activity}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-stone-200 dark:stroke-stone-800" stroke="" vertical={false} />
              <XAxis dataKey="time" className="text-stone-500 dark:text-stone-400" fontSize={12} tickLine={false} axisLine={false} />
              <YAxis className="text-stone-500 dark:text-stone-400" fontSize={12} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "8px" }} />
              <Line type="monotone" dataKey="events" stroke="var(--chart-1)" strokeWidth={2} dot={false} activeDot={{ fill: 'var(--chart-1)', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)', padding: '16px', boxShadow: 'var(--shadow-sm)', display: 'flex', flexDirection: 'column' }}>
        <h3 className="text-base font-medium text-stone-900 dark:text-stone-200 tracking-tight mb-4">Color Distribution (last 6h)</h3>
        {!loading && colors.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-stone-500 dark:text-stone-400 gap-2 min-h-[250px]">
            <PieIcon className="w-8 h-8 opacity-50" />
            <span className="text-sm">No data available for selected time range</span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={colors}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={80}
                dataKey="value"
              >
                {colors.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill || "#06b6d4"} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "8px" }} />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>

      <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)', padding: '16px', boxShadow: 'var(--shadow-sm)', display: 'flex', flexDirection: 'column' }}>
        <h3 className="text-base font-medium text-stone-900 dark:text-stone-200 tracking-tight mb-4">Camera Running Status</h3>
        {!loading && perf.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-stone-500 dark:text-stone-400 gap-2 min-h-[250px]">
            <BarChart3 className="w-8 h-8 opacity-50" />
            <span className="text-sm">No data available for selected time range</span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={perf}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-stone-200 dark:stroke-stone-800" stroke="" vertical={false} />
              <XAxis dataKey="camera" className="text-stone-500 dark:text-stone-400" fontSize={12} tickLine={false} axisLine={false} angle={-45} textAnchor="end" height={80} />
              <YAxis className="text-stone-500 dark:text-stone-400" fontSize={12} tickLine={false} axisLine={false} domain={[0, 100]} />
              <Tooltip cursor={{ fill: "rgba(255,255,255,0.05)" }} contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "8px" }} />
              <Bar dataKey="uptime" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      <div style={{ background: 'var(--color-surface)', border: '1px solid var(--color-border)', borderRadius: 'var(--radius-lg)', padding: '16px', boxShadow: 'var(--shadow-sm)', display: 'flex', flexDirection: 'column' }}>
        <h3 className="text-base font-medium text-stone-900 dark:text-stone-200 tracking-tight mb-4">Media Items</h3>
        {!loading && media.reduce((acc, curr) => acc + curr.value, 0) === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-stone-500 dark:text-stone-400 gap-2 min-h-[250px]">
            <Database className="w-8 h-8 opacity-50" />
            <span className="text-sm">No data available for selected time range</span>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={media}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, value }) => `${name}: ${value}`}
                outerRadius={80}
                dataKey="value"
              >
                <Cell fill="var(--chart-1)" />
                <Cell fill="#27272a" />
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: "8px" }} />
            </PieChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
