"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { AlertsHeader } from "@/components/alerts/alerts-header"
import { AlertsTable } from "@/components/alerts/alerts-table"
import { AlertsMap } from "@/components/alerts/alerts-map"
import { useEffect, useState } from "react"
import { api } from "@/lib/api"
import { toast } from "@/hooks/use-toast"

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
      <div className="space-y-6">
        <AlertsHeader onAddAlert={() => setShowModal(true)} />
        <div className="grid grid-cols-3 gap-6">
          <div className="col-span-2">
            <AlertsTable />
          </div>
          <div>
            <AlertsMap />
          </div>
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
  const [zoneId, setZoneId] = useState<string>("")
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

  // Load zones for selected cameras (or all) for zone picker
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
        rule.time_of_day = { start: "22:00", end: "06:00", tz: "Asia/Kolkata" }
        rule.event = "class_enter_during_time"
      }
      if (cameraIds.length > 0) rule.cameras = cameraIds
      if (objectName) rule.objects = [{ name: objectName }]
      if (color) rule.color = color
      if (countThreshold && countThreshold > 0) rule.count = { ">=": Number(countThreshold) }
      if (zoneId) rule.area = { zone_id: zoneId }
      if (occupancyPct > 0) rule.occupancy_pct = { ">=": Number(occupancyPct) }
      if ((nl || name).toLowerCase().includes("fight")) rule.behavior = "fight"

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
      // notify other components to refresh
      document.dispatchEvent(new CustomEvent("alerts:refresh"))
      onClose()
    } catch (e: any) {
      setError(e?.message || "Failed to create alert")
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <Card className="max-w-md w-full">
        <CardHeader>
          <CardTitle>Create Alert</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && <p className="text-sm text-destructive">{error}</p>}

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Name</label>
              <input
                type="text"
                placeholder="My Alert"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
              />
            </div>

            <div className="flex flex-wrap gap-2 pt-1">
              <span className="text-xs text-muted-foreground pr-2">Quick templates:</span>
              <button
                type="button"
                className="px-2 py-1 text-xs rounded border border-border hover:bg-accent/10"
                onClick={() => {
                  setName("Crowd > 3 people");
                  setObjectName("person");
                  setCountThreshold(3);
                  setZoneId("");
                  setOccupancyPct(0);
                  setSeverity("warning");
                  setCooldownSec(120);
                  setNightHours(false);
                }}
              >
                Crowd &gt; 3
              </button>
              <button
                type="button"
                className="px-2 py-1 text-xs rounded border border-border hover:bg-accent/10"
                onClick={() => {
                  setName("Car enters at night");
                  setObjectName("car");
                  setCountThreshold(1);
                  setSeverity("warning");
                  setCooldownSec(300);
                  setNightHours(true);
                }}
              >
                Car at night
              </button>
              <button
                type="button"
                className="px-2 py-1 text-xs rounded border border-border hover:bg-accent/10"
                onClick={() => {
                  setName("Possible fight");
                  setObjectName("person");
                  setCountThreshold(0);
                  setSeverity("critical");
                  setCooldownSec(180);
                  setNightHours(false);
                  setNl("People fighting");
                }}
              >
                Fight
              </button>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Natural Language (optional)</label>
              <input
                type="text"
                placeholder="People wearing red backpacks after 6 PM"
                value={nl}
                onChange={(e) => setNl(e.target.value)}
                className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Time window (minutes)</label>
                <input
                  type="number"
                  min={1}
                  value={lastMinutes}
                  onChange={(e) => setLastMinutes(parseInt(e.target.value || "0", 10))}
                  className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
                />
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Object</label>
                <input
                  type="text"
                  placeholder="person"
                  value={objectName}
                  onChange={(e) => setObjectName(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Color (optional)</label>
                <input
                  type="text"
                  placeholder="Red"
                  value={color}
                  onChange={(e) => setColor(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
                />
              </div>
              <div className="flex items-center gap-2 pt-6">
                <input id="enabled" type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
                <label htmlFor="enabled" className="text-sm text-foreground">
                  Enabled
                </label>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Count threshold (&gt;=)</label>
                <input
                  type="number"
                  min={0}
                  value={countThreshold}
                  onChange={(e) => setCountThreshold(parseInt(e.target.value || "0", 10))}
                  className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
                />
              </div>
              <div className="flex items-center gap-2 pt-6">
                <input id="nightHours" type="checkbox" checked={nightHours} onChange={(e) => setNightHours(e.target.checked)} />
                <label htmlFor="nightHours" className="text-sm text-foreground">
                  Night hours 22:00–06:00
                </label>
              </div>
            </div>

            {zonesByCamera.length > 0 && (
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Zone (optional)</label>
                  <select
                    value={zoneId}
                    onChange={(e) => setZoneId(e.target.value)}
                    className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
                  >
                    <option value="">Full frame</option>
                    {zonesByCamera.map((z) => (
                      <option key={`${z.camera_id}-${z.zone_id}`} value={z.zone_id}>
                        {z.name} (cam {z.camera_id})
                      </option>
                    ))}
                  </select>
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium text-foreground">Occupancy % (&gt;=)</label>
                  <input
                    type="number"
                    min={0}
                    max={100}
                    value={occupancyPct}
                    onChange={(e) => setOccupancyPct(parseInt(e.target.value || "0", 10))}
                    className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
                    placeholder="0"
                  />
                </div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Severity</label>
                <select
                  value={severity}
                  onChange={(e) => setSeverity(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
                >
                  <option value="info">info</option>
                  <option value="warning">warning</option>
                  <option value="critical">critical</option>
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Cooldown (sec)</label>
                <input
                  type="number"
                  min={0}
                  value={cooldownSec}
                  onChange={(e) => setCooldownSec(parseInt(e.target.value || "0", 10))}
                  className="w-full px-3 py-2 text-sm border border-input rounded-md bg-background"
                />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Cameras (optional)</label>
              <div className="flex flex-wrap gap-2">
                {loadingCameras ? (
                  <span className="text-xs text-muted-foreground">Loading cameras...</span>
                ) : cameras.length === 0 ? (
                  <span className="text-xs text-muted-foreground">No cameras found</span>
                ) : (
                  cameras.map((c) => {
                    const selected = cameraIds.includes(c.camera_id)
                    return (
                      <button
                        key={c.camera_id}
                        type="button"
                        onClick={() => toggleCamera(c.camera_id)}
                        className={`px-3 py-1 rounded-full text-xs border transition-colors cursor-pointer ${
                          selected
                            ? "bg-primary/20 border-primary text-primary"
                            : "bg-muted border-border text-muted-foreground hover:bg-muted/80"
                        }`}
                      >
                        #{c.camera_id} {c.location ? `· ${c.location}` : ""}
                      </button>
                    )
                  })
                )}
              </div>
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <Button variant="outline" onClick={onClose} disabled={saving} className="flex-1">
              Cancel
            </Button>
            <Button onClick={onCreate} disabled={saving} className="flex-1">
              {saving ? "Creating..." : "Create Alert"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
