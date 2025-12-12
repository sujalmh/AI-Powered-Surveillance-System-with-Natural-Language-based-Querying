"use client";

import { MoreVertical, Trash2, Edit2, RefreshCw, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";

type AlertItem = {
  id: string;
  name: string;
  nl?: string;
  rule: Record<string, any>;
  enabled: boolean;
  actions: string[];
  created_at: string;
  updated_at: string;
};

export function AlertsTable() {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const [evaluatingId, setEvaluatingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.listAlerts();
      // normalize id field to string
      const list: AlertItem[] = (data as any[]).map((a) => ({
        id: String(a.id),
        name: a.name,
        nl: a.nl,
        rule: a.rule ?? {},
        enabled: Boolean(a.enabled),
        actions: a.actions ?? [],
        created_at: a.created_at ?? "",
        updated_at: a.updated_at ?? "",
      }));
      setAlerts(list);
    } catch (e: any) {
      setError(e?.message || "Failed to load alerts");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  // Listen for external refresh requests (e.g., after creating an alert)
  useEffect(() => {
    const handler = () => load();
    // CustomEvent name: 'alerts:refresh'
    document.addEventListener("alerts:refresh", handler as any);
    return () => document.removeEventListener("alerts:refresh", handler as any);
  }, []);

  const onDelete = async (id: string) => {
    setDeletingId(id);
    setError(null);
    try {
      await api.deleteAlert(id);
      setAlerts((prev: AlertItem[]) => prev.filter((a: AlertItem) => a.id !== id));
    } catch (e: any) {
      setError(e?.message || "Delete failed");
    } finally {
      setDeletingId(null);
      setOpenMenu(null);
    }
  };

  const onEvaluate = async (id: string) => {
    setEvaluatingId(id);
    setError(null);
    try {
      await api.evaluateAlert(id);
    } catch (e: any) {
      setError(e?.message || "Evaluate failed");
    } finally {
      setEvaluatingId(null);
      setOpenMenu(null);
    }
  };

  type Row = AlertItem & { condition?: string };
  const rows: Row[] = useMemo(() => {
    return alerts.map((a: AlertItem) => {
      const rule = a.rule || {};
      const obj = Array.isArray(rule.objects) && rule.objects.length > 0 ? rule.objects[0]?.name : undefined;
      const color = rule.color;
      const time =
        rule?.time?.last_minutes
          ? `last ${rule.time.last_minutes}m`
          : rule?.time?.last_hours
          ? `last ${rule.time.last_hours}h`
          : rule?.time?.from || rule?.time?.to
          ? `${rule.time.from || ""} ${rule.time.to || ""}`.trim()
          : "";
      const condParts: string[] = [];
      if (obj) condParts.push(obj);
      if (color) condParts.push(color);
      if (time) condParts.push(`(${time})`);
      const condition = condParts.join(" ");
      return {
        ...a,
        condition,
      };
    });
  }, [alerts]);

  return (
    <div className="bg-card border border-border rounded-2xl p-6 overflow-x-auto shadow-[0_2px_10px_rgba(25,24,59,0.1)]">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-card-foreground">Configured Alerts</h2>
          {loading && <span className="text-xs text-muted-foreground">Loading…</span>}
          {error && <span className="text-xs text-destructive">{error}</span>}
        </div>
        <button
          onClick={load}
          className="px-3 py-1.5 rounded-lg bg-accent/10 border border-accent text-xs text-accent-foreground flex items-center gap-2 hover:bg-accent/20 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Refresh
        </button>
      </div>

      <table className="w-full">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-3 px-4 text-sm font-semibold text-muted-foreground">Name</th>
            <th className="text-left py-3 px-4 text-sm font-semibold text-muted-foreground">Condition</th>
            <th className="text-left py-3 px-4 text-sm font-semibold text-muted-foreground">Enabled</th>
            <th className="text-left py-3 px-4 text-sm font-semibold text-muted-foreground">Updated</th>
            <th className="text-left py-3 px-4 text-sm font-semibold text-muted-foreground">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((alert: Row) => (
            <tr key={alert.id} className="border-b border-border/50 hover:bg-accent/5 transition-colors">
              <td className="py-3 px-4 text-sm text-card-foreground font-medium">{alert.name}</td>
              <td className="py-3 px-4 text-sm text-muted-foreground">
                {alert.condition || alert.nl || <span className="italic text-muted-foreground">n/a</span>}
              </td>
              <td className="py-3 px-4 text-sm">
                <span
                  className={`px-3 py-1 rounded-full text-xs font-semibold ${
                    alert.enabled
                      ? "bg-accent/20 text-accent border border-accent"
                      : "bg-muted/20 text-muted-foreground border border-border"
                  }`}
                >
                  {alert.enabled ? "Enabled" : "Disabled"}
                </span>
              </td>
              <td className="py-3 px-4 text-xs text-muted-foreground">
                {alert.updated_at ? new Date(alert.updated_at).toLocaleString() : "-"}
              </td>
              <td className="py-3 px-4 text-sm">
                <div className="relative">
                  <button
                    onClick={() => setOpenMenu(openMenu === alert.id ? null : alert.id)}
                    className="p-1 hover:bg-accent/10 rounded transition-colors"
                  >
                    <MoreVertical className="w-4 h-4 text-muted-foreground" />
                  </button>
                  {openMenu === alert.id && (
                    <div className="absolute right-0 top-full mt-1 bg-card border border-border rounded-2xl p-2 space-y-1 z-10 min-w-[150px] shadow-[0_4px_12px_rgba(25,24,59,0.15)]">
                      <button
                        disabled={evaluatingId === alert.id}
                        onClick={() => onEvaluate(alert.id)}
                        className="flex items-center gap-2 px-3 py-2 text-sm text-card-foreground hover:bg-accent/10 rounded w-full disabled:opacity-60 transition-colors"
                      >
                        <Play className="w-4 h-4" />
                        {evaluatingId === alert.id ? "Evaluating…" : "Evaluate"}
                      </button>
                      <button
                        className="flex items-center gap-2 px-3 py-2 text-sm text-card-foreground hover:bg-accent/10 rounded w-full transition-colors"
                        disabled
                        title="Edit not implemented"
                      >
                        <Edit2 className="w-4 h-4" />
                        Edit
                      </button>
                      <button
                        disabled={deletingId === alert.id}
                        onClick={() => onDelete(alert.id)}
                        className="flex items-center gap-2 px-3 py-2 text-sm text-destructive hover:bg-destructive/10 rounded w-full disabled:opacity-60 transition-colors"
                      >
                        <Trash2 className="w-4 h-4" />
                        {deletingId === alert.id ? "Deleting…" : "Delete"}
                      </button>
                    </div>
                  )}
                </div>
              </td>
            </tr>
          ))}
          {rows.length === 0 && !loading && (
            <tr>
              <td colSpan={5} className="py-6 text-center text-sm text-muted-foreground">
                No alerts configured yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
