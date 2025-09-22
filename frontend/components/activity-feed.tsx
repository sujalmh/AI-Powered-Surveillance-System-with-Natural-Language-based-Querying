"use client"

import { useState } from "react"
import { Bell, Car, Clock, Package, User, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"

type ActivityItem = {
  id: string
  type: "person" | "vehicle" | "object" | "alert"
  title: string
  location: string
  time: string
  description: string
}

const activityData: ActivityItem[] = [
  {
    id: "1",
    type: "person",
    title: "Person Detected",
    location: "Main Entrance",
    time: "2 minutes ago",
    description: "Individual entered through main door",
  },
  {
    id: "2",
    type: "vehicle",
    title: "Vehicle Entered",
    location: "Parking Lot",
    time: "15 minutes ago",
    description: "Blue sedan parked in spot A12",
  },
  {
    id: "3",
    type: "alert",
    title: "Unusual Activity",
    location: "Storage Room",
    time: "32 minutes ago",
    description: "Movement detected after hours",
  },
  {
    id: "4",
    type: "object",
    title: "Package Detected",
    location: "Reception",
    time: "1 hour ago",
    description: "Package left at front desk",
  },
  {
    id: "5",
    type: "person",
    title: "Multiple People",
    location: "Conference Room",
    time: "1.5 hours ago",
    description: "Group of 5 people entered meeting room",
  },
  {
    id: "6",
    type: "alert",
    title: "Restricted Area Access",
    location: "Server Room",
    time: "2 hours ago",
    description: "Unauthorized access attempt",
  },
  {
    id: "7",
    type: "vehicle",
    title: "Vehicle Exit",
    location: "Parking Lot",
    time: "2.5 hours ago",
    description: "Red truck left premises",
  },
]

export function ActivityFeed() {
  const [activities] = useState<ActivityItem[]>(activityData)

  const getIcon = (type: ActivityItem["type"]) => {
    switch (type) {
      case "person":
        return <User className="h-4 w-4" />
      case "vehicle":
        return <Car className="h-4 w-4" />
      case "object":
        return <Package className="h-4 w-4" />
      case "alert":
        return <AlertTriangle className="h-4 w-4" />
      default:
        return <Bell className="h-4 w-4" />
    }
  }

  const getIconBackground = (type: ActivityItem["type"]) => {
    switch (type) {
      case "person":
        return "bg-blue-100 text-blue-700 dark:bg-blue-900/20"
      case "vehicle":
        return "bg-green-100 text-green-700 dark:bg-green-900/20"
      case "object":
        return "bg-purple-100 text-purple-700 dark:bg-purple-900/20"
      case "alert":
        return "bg-red-100 text-red-700 dark:bg-red-900/20"
      default:
        return "bg-gray-100 text-gray-700 dark:bg-gray-800"
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <ScrollArea className="h-[300px] pr-4">
        <div className="space-y-4">
          {activities.map((activity) => (
            <div key={activity.id} className="flex items-start gap-4 rounded-lg border p-3">
              <div
                className={`flex h-8 w-8 items-center justify-center rounded-full ${getIconBackground(activity.type)}`}
              >
                {getIcon(activity.type)}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold">{activity.title}</h4>
                </div>
                <p className="text-sm text-muted-foreground">{activity.location}</p>
                <p className="text-sm">{activity.description}</p>
                <div className="mt-1 flex items-center text-xs text-muted-foreground">
                  <Clock className="mr-1 h-3 w-3" />
                  <span>{activity.time}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </ScrollArea>
      <Button variant="outline" size="sm" className="mt-2">
        View All Activity
      </Button>
    </div>
  )
}
