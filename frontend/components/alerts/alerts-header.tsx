"use client"

import { Plus } from "lucide-react"
import { Button } from "@/components/ui/button"

interface AlertsHeaderProps {
  onAddAlert: () => void
}

export function AlertsHeader({ onAddAlert }: AlertsHeaderProps) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <h1 className="text-4xl font-bold text-foreground mb-2">Alerts</h1>
        <p className="text-muted-foreground">Manage and monitor system alerts</p>
      </div>
      <Button onClick={onAddAlert}>
        <Plus className="w-5 h-5" />
        Add Alert
      </Button>
    </div>
  )
}
