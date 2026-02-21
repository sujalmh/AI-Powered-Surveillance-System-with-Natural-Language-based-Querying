"use client";

import { useMemo, useState, useEffect } from "react";
import useSWR from "swr";
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { MoreVertical, Trash2, Play, Edit2, Plus } from "lucide-react";

import { api } from "@/lib/api";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/hooks/use-toast";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

type AlertItem = {
  id: string;
  name: string;
  nl?: string;
  rule: Record<string, any>;
  enabled: boolean;
  actions: string[];
  created_at: string;
  updated_at: string;
  condition?: string;
};

const fetchAlerts = async (): Promise<AlertItem[]> => {
  const data = await api.listAlerts();
  return (data as any[]).map((a) => {
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
    const count = rule?.count;
    if (count && (count[">="] ?? count[">"])) condParts.push(`≥${count[">="] ?? count[">"]}`);
    if (time) condParts.push(`(${time})`);
    
    return {
      id: String(a.id),
      name: a.name,
      nl: a.nl,
      rule: a.rule ?? {},
      enabled: Boolean(a.enabled),
      actions: a.actions ?? [],
      created_at: a.created_at ?? "",
      updated_at: a.updated_at ?? "",
      condition: condParts.join(" "),
    };
  });
};

export function AlertsTable({ onAddAlert }: { onAddAlert: () => void }) {
  const { data: alerts, error, isLoading, mutate } = useSWR("alerts", fetchAlerts, { refreshInterval: 10000 });

  useEffect(() => {
    const handler = () => mutate();
    document.addEventListener("alerts:refresh", handler as any);
    return () => document.removeEventListener("alerts:refresh", handler as any);
  }, [mutate]);

  const onDelete = async (id: string) => {
    try {
      await api.deleteAlert(id);
      toast({ title: "Alert deleted", description: "Successfully removed alert rule." });
      mutate();
    } catch (e: any) {
      toast({ title: "Delete Failed", description: e?.message || "Could not delete alert.", variant: "destructive" });
    }
  };

  const onEvaluate = async (id: string) => {
    try {
      toast({ title: "Evaluating...", description: "Testing the alert rule..." });
      await api.evaluateAlert(id);
      toast({ title: "Evaluate Complete", description: "Triggered active evaluation cycle." });
    } catch (e: any) {
      toast({ title: "Evaluation Failed", description: e?.message || "Failed to test alert.", variant: "destructive" });
    }
  };

  const columns = useMemo<ColumnDef<AlertItem>[]>(() => [
    {
      accessorKey: "name",
      header: "Name",
      cell: ({ row }) => <span className="font-medium">{row.getValue("name")}</span>,
    },
    {
      accessorKey: "condition",
      header: "Condition",
      cell: ({ row }) => (
        <span className="text-muted-foreground text-sm">
          {row.getValue("condition") || row.original.nl || <span className="italic">n/a</span>}
        </span>
      ),
    },
    {
      accessorKey: "enabled",
      header: "Status",
      cell: ({ row }) => (
        <Switch
          checked={row.getValue("enabled")}
          onCheckedChange={() => toast({ title: "Not Supported", description: "Edit functionality not implemented on backend." })}
        />
      ),
    },
    {
      accessorKey: "updated_at",
      header: "Updated",
      cell: ({ row }) => (
        <span className="text-muted-foreground text-xs">
          {row.getValue("updated_at") ? new Date(row.getValue("updated_at") as string).toLocaleString() : "-"}
        </span>
      ),
    },
    {
      id: "actions",
      cell: ({ row }) => {
        const id = row.original.id;
        return (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-8 w-8 p-0">
                <span className="sr-only">Open menu</span>
                <MoreVertical className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-[160px]">
              <DropdownMenuItem onClick={() => onEvaluate(id)}>
                <Play className="mr-2 h-4 w-4" /> Evaluate
              </DropdownMenuItem>
              <DropdownMenuItem disabled>
                <Edit2 className="mr-2 h-4 w-4" /> Edit
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onDelete(id)} className="text-rose-500 focus:text-rose-500">
                <Trash2 className="mr-2 h-4 w-4" /> Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        );
      },
    },
  ], []);

  const table = useReactTable({
    data: alerts || [],
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <Card className="shadow-sm border-border bg-card overflow-hidden">
      <CardHeader className="pb-4 flex flex-row items-center justify-between shadow-none">
        <CardTitle className="text-lg">Configured Alerts</CardTitle>
        <div className="flex items-center gap-4">
          {isLoading && !alerts && <span className="text-xs text-muted-foreground animate-pulse">Loading...</span>}
          <Button onClick={onAddAlert} className="rounded-full h-8 px-4 text-xs font-semibold shadow-sm">
            <Plus className="w-3.5 h-3.5 mr-1" />
            Add Alert
          </Button>
        </div>
      </CardHeader>
      <CardContent className="p-0 border-t border-border">
        {error ? (
          <div className="p-6 text-sm text-rose-500">Failed to load alerts: {error.message}</div>
        ) : (
          <Table>
            <TableHeader className="bg-muted/40 hover:bg-muted/40">
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <TableHead key={header.id} className="text-xs uppercase tracking-wider font-semibold text-muted-foreground">
                      {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                    </TableHead>
                  ))}
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows?.length ? (
                table.getRowModel().rows.map((row) => (
                  <TableRow key={row.id} data-state={row.getIsSelected() && "selected"}>
                    {row.getVisibleCells().map((cell) => (
                      <TableCell key={cell.id} className="py-3 px-4">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    ))}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={columns.length} className="h-24 text-center text-sm text-muted-foreground">
                    {isLoading ? "Loading rules..." : "No alert rules configured."}
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
