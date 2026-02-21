"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { Badge } from "@/components/ui/badge"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Plus } from "lucide-react"
import { AlertsTable } from "@/components/alerts/alerts-table"
import { AlertsMap } from "@/components/alerts/alerts-map"
import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import { toast } from "@/hooks/use-toast"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

export default function AlertsPage() {
  const [showModal, setShowModal] = useState(false)

  useEffect(() => {
    const es = api.streamAlerts();
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        toast({
          title: data.message || "Alert triggered",
          description: `Camera #${data.camera_id ?? "-"} · ${data.triggered_at ? new Date(data.triggered_at).toLocaleString() : ""}`,
        });
      } catch (e) {
        // ignore parse errors
      }
    };
    return () => es.close();
  }, [])

  return (
    <MainLayout>
      {/* Clean, Professional Header Area */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-foreground mb-2 flex items-center gap-3">
          Security Alerts
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-primary/10 text-primary text-xs font-medium uppercase tracking-wider">
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
            Active
          </span>
        </h1>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <p className="text-muted-foreground text-sm max-w-xl">
            Manage natural language triggers and monitor your entire camera fleet in real time.
          </p>
          <Button 
            onClick={() => setShowModal(true)}
            className="shadow-sm"
          >
            <Plus className="w-4 h-4 mr-2" />
            Create Alert
          </Button>
        </div>
      </div>

      <div className="space-y-6">
        {/* Stacked Layout instead of grid */}
        <div className="w-full">
          <AlertsTable />
        </div>
        <div className="w-full">
          <AlertsMap />
        </div>
        
        {showModal && <AlertModal onClose={() => setShowModal(false)} />}
      </div>
    </MainLayout>
  )
}

function AlertModal({ onClose }: { onClose: () => void }) {
  const [name, setName] = useState("")
  const [nl, setNl] = useState("")
  const [lastMinutes, setLastMinutes] = useState<number>(60)
  const [objectName, setObjectName] = useState<string>("person")
  const [color, setColor] = useState<string>("")
  const [countThreshold, setCountThreshold] = useState<number>(0)
  const [zoneId, setZoneId] = useState<string>("_all")
  const [occupancyPct, setOccupancyPct] = useState<number>(0)
  const [severity, setSeverity] = useState<string>("info")
  const [cooldownSec, setCooldownSec] = useState<number>(60)
  const [nightHours, setNightHours] = useState<boolean>(false)
  const [enabled, setEnabled] = useState<boolean>(true)
  const [cameraIds, setCameraIds] = useState<number[]>([])
  const [cameras, setCameras] = useState<Array<{ camera_id: number; location?: string }>>([])
  const [zonesByCamera, setZonesByCamera] = useState<Array<{ camera_id: number; zone_id: string; name: string }>>([])
  const [loadingCameras, setLoadingCameras] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true
    setLoadingCameras(true)
    setError(null)
    api
      .listCameras()
      .then((list) => {
        if (!mounted) return
        const cams = (list as any[]).map((c) => ({ camera_id: Number(c.camera_id), location: c.location }))
        setCameras(cams)
        setLoadingCameras(false)
      })
      .catch((e) => {
        if (!mounted) return
        setError(e?.message || "Failed to load cameras")
        setLoadingCameras(false)
      })
    return () => {
      mounted = false
    }
  }, [])

  useEffect(() => {
    if (cameras.length === 0) return
    let mounted = true
    const ids = cameraIds.length > 0 ? cameraIds : cameras.map((c) => c.camera_id)
    const all: Array<{ camera_id: number; zone_id: string; name: string }> = []
    Promise.all(
      ids.map((cid) =>
        api.listZones(cid).then((zones: any[]) => {
          if (!mounted) return
          zones.forEach((z: any) => all.push({ camera_id: cid, zone_id: z.zone_id, name: z.name || z.zone_id }))
        }).catch(() => {})
      )
    ).then(() => {
      if (mounted) setZonesByCamera(all)
    })
    return () => { mounted = false }
  }, [cameraIds, cameras])

  const toggleCamera = (id: number) => {
    setCameraIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
  }

  const onCreate = async () => {
    setSaving(true)
    setError(null)
    try {
      const rule: any = {
        time: { last_minutes: lastMinutes },
      }
      if (nightHours) {
        const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
        rule.time_of_day = { start: "22:00", end: "06:00", tz }
        rule.event = "class_enter_during_time"
      }
      if (cameraIds.length > 0) rule.cameras = cameraIds
      if (objectName) rule.objects = [{ name: objectName }]
      if (color) rule.color = color
      if (countThreshold && countThreshold > 0) rule.count = { ">=": Number(countThreshold) }
      if (zoneId && zoneId !== "_all") rule.area = { zone_id: zoneId }
      if (occupancyPct > 0) rule.occupancy_pct = { ">=": Number(occupancyPct) }
      if ((nl && nl.toLowerCase().includes("fight")) || (name && name.toLowerCase().includes("fight"))) rule.behavior = "fight"

      const payload = {
        name: name || "New Alert",
        nl: nl || undefined,
        rule,
        enabled,
        actions: ["push_ws", "snapshot"],
        severity,
        cooldown_sec: Number(cooldownSec),
      }
      await api.createAlert(payload)
      document.dispatchEvent(new CustomEvent("alerts:refresh"))
      onClose()
    } catch (e: any) {
      setError(e?.message || "Failed to create alert")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-200">
      <Card className="max-w-2xl w-full max-h-[90vh] flex flex-col shadow-lg border border-border bg-card overflow-hidden animate-in zoom-in-95 duration-200">
        <CardHeader className="border-b border-border py-4 px-6 shrink-0 bg-muted/20">
          <CardTitle className="text-lg font-semibold">Create Alert Rule</CardTitle>
          <p className="text-sm text-muted-foreground mt-1">Configure natural language triggers or explicit parameters.</p>
        </CardHeader>
        <CardContent className="overflow-y-auto flex-1 p-6 space-y-6">
          {error && (
             <div className="p-3 bg-rose-500/10 border border-rose-500/20 text-rose-500 rounded-lg text-sm">
               {error}
             </div>
          )}

          <div className="space-y-5">
            <div className="space-y-2">
              <Label>Name</Label>
              <Input
                placeholder="My Alert..."
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            <div className="flex flex-wrap items-center gap-2 bg-muted/30 p-3 rounded-xl border border-border/50">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mr-2 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-primary/70"></span>
                Templates
              </span>
              <Button
                variant="outline"
                size="sm"
                className="h-8 text-xs px-3 rounded-full hover:bg-green-500/10 hover:text-green-600 hover:border-green-500/30 transition-colors"
                onClick={() => {
                  setName("Crowd > 3 people")
                  setObjectName("person")
                  setCountThreshold(3)
                  setZoneId("_all")
                  setOccupancyPct(0)
                  setSeverity("warning")
                  setCooldownSec(120)
                  setNightHours(false)
                }}
              >
                Crowd &gt; 3
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-8 text-xs px-3 rounded-full hover:bg-blue-500/10 hover:text-blue-600 hover:border-blue-500/30 transition-colors"
                onClick={() => {
                  setName("Car enters at night")
                  setObjectName("car")
                  setCountThreshold(1)
                  setSeverity("warning")
                  setCooldownSec(300)
                  setNightHours(true)
                }}
              >
                Car at night
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-8 text-xs px-3 rounded-full hover:bg-rose-500/10 hover:text-rose-600 hover:border-rose-500/30 transition-colors"
                onClick={() => {
                  setName("Possible fight")
                  setObjectName("person")
                  setCountThreshold(0)
                  setSeverity("critical")
                  setCooldownSec(180)
                  setNightHours(false)
                  setNl("People fighting")
                }}
              >
                Fight
              </Button>
            </div>

            <div className="space-y-2">
              <Label>Natural Language (optional)</Label>
              <Input
                placeholder="E.g. People wearing red backpacks after 6 PM"
                value={nl}
                onChange={(e) => setNl(e.target.value)}
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Time window (minutes)</Label>
                <Input
                  type="number"
                  min={1}
                  value={lastMinutes}
                  onChange={(e) => setLastMinutes(parseInt(e.target.value || "0", 10))}
                />
              </div>
              <div className="space-y-2">
                <Label>Object Class</Label>
                <Input
                  placeholder="person, car, truck..."
                  value={objectName}
                  onChange={(e) => setObjectName(e.target.value)}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Color (optional)</Label>
                <Input
                  placeholder="Red, Blue..."
                  value={color}
                  onChange={(e) => setColor(e.target.value)}
                />
              </div>
              <div className="flex items-center gap-2 pt-8">
                <Checkbox 
                  id="enabled" 
                  checked={enabled} 
                  onCheckedChange={(checked) => setEnabled(checked as boolean)} 
                />
                <Label htmlFor="enabled">Enabled</Label>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Count threshold (&gt;=)</Label>
                <Input
                  type="number"
                  min={0}
                  value={countThreshold}
                  onChange={(e) => setCountThreshold(parseInt(e.target.value || "0", 10))}
                />
              </div>
              <div className="flex items-center gap-2 pt-8">
                <Checkbox 
                  id="nightHours" 
                  checked={nightHours} 
                  onCheckedChange={(checked) => setNightHours(checked as boolean)} 
                />
                <Label htmlFor="nightHours">Night hours (22:00–06:00)</Label>
              </div>
            </div>

            {zonesByCamera.length > 0 && (
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Zone Restriction</Label>
                  <Select value={zoneId} onValueChange={setZoneId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Full frame" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="_all">Full frame</SelectItem>
                      {zonesByCamera.map((z) => (
                        <SelectItem key={`${z.camera_id}-${z.zone_id}`} value={z.zone_id}>
                          {z.name} (cam {z.camera_id})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Occupancy % (&gt;=)</Label>
                  <Input
                    type="number"
                    min={0}
                    max={100}
                    value={occupancyPct}
                    onChange={(e) => setOccupancyPct(parseInt(e.target.value || "0", 10))}
                  />
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Severity</Label>
                <Select value={severity} onValueChange={setSeverity}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="info">Info</SelectItem>
                    <SelectItem value="warning">Warning</SelectItem>
                    <SelectItem value="critical">Critical</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Cooldown (sec)</Label>
                <Input
                  type="number"
                  min={0}
                  value={cooldownSec}
                  onChange={(e) => setCooldownSec(parseInt(e.target.value || "0", 10))}
                />
              </div>
            </div>

            <div className="space-y-3 pt-4 border-t border-border/50">
              <Label className="text-muted-foreground">Cameras (optional, leaving empty applies to all)</Label>
              <div className="flex flex-wrap gap-2">
                {loadingCameras ? (
                  <span className="text-xs text-muted-foreground animate-pulse">Loading cameras...</span>
                ) : cameras.length === 0 ? (
                  <span className="text-xs text-muted-foreground">No cameras found</span>
                ) : (
                  cameras.map((c) => {
                    const selected = cameraIds.includes(c.camera_id)
                    return (
                      <Badge
                        key={c.camera_id}
                        variant={selected ? "default" : "outline"}
                        role="button"
                        tabIndex={0}
                        aria-pressed={selected}
                        className={`cursor-pointer px-3 py-1 text-xs transition-all ${
                          selected 
                            ? "bg-green-500/20 text-green-700 dark:text-green-400 border-green-500/50 hover:bg-green-500/30" 
                            : "hover:bg-muted font-normal text-muted-foreground"
                        }`}
                        onClick={() => toggleCamera(c.camera_id)}
                        onKeyDown={(e: React.KeyboardEvent) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault()
                            toggleCamera(c.camera_id)
                          }
                        }}
                      >
                        #{c.camera_id} {c.location ? `· ${c.location}` : ""}
                      </Badge>
                    )
                  })
                )}
              </div>
            </div>
          </div>
        </CardContent>
        <div className="p-4 px-6 border-t border-border bg-muted/20 shrink-0 flex gap-3 justify-end items-center">
          <Button variant="outline" onClick={onClose} disabled={saving}>
            Cancel
          </Button>
          <Button onClick={onCreate} disabled={saving}>
            {saving ? "Creating..." : "Create Alert"}
          </Button>
        </div>
      </Card>
    </div>
  )
}
