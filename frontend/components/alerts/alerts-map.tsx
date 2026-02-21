"use client";

import { useMemo, useEffect } from "react";
import useSWR from "swr";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

type CameraDoc = {
  camera_id: number;
  source?: string;
  location?: string;
  status?: string;
  last_seen?: string;
  running?: boolean;
};

type AlertLog = {
  alert_id: string;
  triggered_at: string;
  camera_id?: number;
  message: string;
  snapshot?: string | null;
  clip?: string | null;
};

const fetchMapData = async () => {
  const [cams, lgs] = await Promise.all([api.listCameras(), api.listAlertLogs()]);
  return {
    cameras: (cams as any[]).map((c) => ({
      camera_id: Number(c.camera_id),
      source: c.source,
      location: c.location,
      status: c.status,
      last_seen: c.last_seen,
      running: Boolean(c.running),
    })),
    logs: lgs as AlertLog[],
  };
};

export function AlertsMap() {
  const { data, error, isLoading, mutate } = useSWR("alertsMapData", fetchMapData, { refreshInterval: 10000 });

  useEffect(() => {
    const handler = () => mutate();
    document.addEventListener("alerts:refresh", handler as any);
    return () => document.removeEventListener("alerts:refresh", handler as any);
  }, [mutate]);

  const alertingCameras = useMemo(() => {
    const set = new Set<number>();
    if (!data?.logs) return set;
    for (const l of data.logs) {
      if (typeof l.camera_id === "number") set.add(l.camera_id);
    }
    return set;
  }, [data?.logs]);

  return (
    <Card className="h-full flex flex-col shadow-sm border-border bg-card">
      <CardHeader className="pb-4 flex flex-row items-center justify-between shadow-none">
        <CardTitle className="text-lg">Camera Status</CardTitle>
        {isLoading && !data && <span className="text-xs text-muted-foreground animate-pulse">Loading...</span>}
      </CardHeader>
      
      <CardContent className="p-0 border-t border-border overflow-x-auto">
        {error ? (
           <div className="p-6 text-sm text-rose-500">Failed to load map data: {error.message}</div>
         ) : (
           <Table>
             <TableHeader className="bg-muted/40 hover:bg-muted/40">
              <TableRow>
                <TableHead className="text-xs uppercase tracking-wider font-semibold text-muted-foreground">Camera</TableHead>
                <TableHead className="text-xs uppercase tracking-wider font-semibold text-muted-foreground">Location</TableHead>
                <TableHead className="text-xs uppercase tracking-wider font-semibold text-muted-foreground">Connection</TableHead>
                <TableHead className="text-xs uppercase tracking-wider font-semibold text-muted-foreground">Alert Status</TableHead>
                <TableHead className="text-xs uppercase tracking-wider font-semibold text-muted-foreground"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.cameras?.length ? (
                data.cameras.map((c) => {
                  const alerting = alertingCameras.has(c.camera_id);
                  return (
                    <TableRow key={c.camera_id}>
                      <TableCell className="font-medium"><span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8125rem', color: 'var(--color-text)' }}>#{c.camera_id}</span></TableCell>
                      <TableCell className="text-muted-foreground">{c.location || "-"}</TableCell>
                      <TableCell>
                        <Badge variant={c.running ? "default" : "secondary"} className={c.running ? 'bg-emerald-500/10 text-emerald-500 border-none hover:bg-emerald-500/20 shadow-none' : 'shadow-none border-none'}>
                          {c.running ? "Running" : "Stopped"}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className={`gap-1.5 pl-1.5 pr-2 ${alerting ? "border-rose-500/50 bg-rose-500/10 text-rose-500" : "border-emerald-500/50 bg-emerald-500/10 text-emerald-500"}`}>
                          <span className={`w-1.5 h-1.5 rounded-full ${alerting ? "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.8)] animate-pulse" : "bg-emerald-500"}`} />
                          {alerting ? "Active Alert" : "Normal"}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {c.last_seen ? new Date(c.last_seen).toLocaleString() : "-"}
                      </TableCell>
                    </TableRow>
                  );
                })
              ) : (
                <TableRow>
                  <TableCell colSpan={5} className="h-24 text-center text-sm text-muted-foreground">
                    {isLoading ? "Fetching camera statuses..." : "No cameras registered yet."}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
