"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import {
  AlertTriangle,
  Bell,
  Clock,
  Filter,
  Search,
  ArrowUpDown,
  Calendar,
  Camera,
  Eye,
  MoreHorizontal,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Calendar as CalendarComponent } from "@/components/ui/calendar"
import { format } from "date-fns"
import { AlertsChart } from "@/components/alerts-chart"

type AlertSeverity = "critical" | "high" | "medium" | "low" | "info"

type Alert = {
  id: string
  title: string
  description: string
  severity: AlertSeverity
  timestamp: Date
  camera: string
  location: string
  status: "new" | "acknowledged" | "resolved"
  videoId?: string
}

const mockAlerts: Alert[] = [
  {
    id: "a1",
    title: "Unauthorized Access",
    description: "Individual entered restricted area without credentials",
    severity: "critical",
    timestamp: new Date(2023, 3, 15, 2, 30),
    camera: "Server Room",
    location: "Basement",
    status: "new",
    videoId: "v7",
  },
  {
    id: "a2",
    title: "Unusual Movement",
    description: "Suspicious activity detected in storage area after hours",
    severity: "high",
    timestamp: new Date(2023, 3, 15, 1, 42),
    camera: "Storage Area",
    location: "Warehouse",
    status: "new",
    videoId: "v6",
  },
  {
    id: "a3",
    title: "After Hours Access",
    description: "Employee badge used outside normal business hours",
    severity: "medium",
    timestamp: new Date(2023, 3, 14, 23, 15),
    camera: "Main Entrance",
    location: "Front Lobby",
    status: "acknowledged",
    videoId: "v1",
  },
  {
    id: "a4",
    title: "Unattended Package",
    description: "Package left unattended in public area",
    severity: "low",
    timestamp: new Date(2023, 3, 14, 16, 20),
    camera: "Lobby",
    location: "Reception",
    status: "resolved",
  },
  {
    id: "a5",
    title: "Tailgating Detected",
    description: "Multiple people entered with single badge scan",
    severity: "high",
    timestamp: new Date(2023, 3, 14, 9, 45),
    camera: "Side Entrance",
    location: "East Wing",
    status: "acknowledged",
    videoId: "v2",
  },
  {
    id: "a6",
    title: "Camera Tampering",
    description: "Possible camera manipulation detected",
    severity: "critical",
    timestamp: new Date(2023, 3, 13, 22, 10),
    camera: "Parking Lot",
    location: "North Side",
    status: "resolved",
    videoId: "v4",
  },
  {
    id: "a7",
    title: "Motion in Secure Zone",
    description: "Movement detected in secure zone after hours",
    severity: "high",
    timestamp: new Date(2023, 3, 13, 1, 15),
    camera: "Executive Office",
    location: "3rd Floor",
    status: "resolved",
  },
  {
    id: "a8",
    title: "Loitering Detected",
    description: "Individual remained in area for extended period",
    severity: "medium",
    timestamp: new Date(2023, 3, 12, 14, 30),
    camera: "Cafeteria",
    location: "1st Floor",
    status: "resolved",
  },
  {
    id: "a9",
    title: "Unusual Behavior",
    description: "Person exhibiting unusual behavior near entrance",
    severity: "medium",
    timestamp: new Date(2023, 3, 12, 11, 20),
    camera: "Main Entrance",
    location: "Front Lobby",
    status: "resolved",
  },
  {
    id: "a10",
    title: "Vehicle in Restricted Area",
    description: "Unauthorized vehicle in restricted parking zone",
    severity: "low",
    timestamp: new Date(2023, 3, 11, 13, 45),
    camera: "Parking Lot",
    location: "North Side",
    status: "resolved",
    videoId: "v5",
  },
]

const getSeverityBadge = (severity: AlertSeverity) => {
  switch (severity) {
    case "critical":
      return <Badge variant="destructive">Critical</Badge>
    case "high":
      return (
        <Badge variant="destructive" className="bg-red-500">
          High
        </Badge>
      )
    case "medium":
      return (
        <Badge variant="outline" className="border-yellow-500 text-yellow-500">
          Medium
        </Badge>
      )
    case "low":
      return <Badge variant="outline">Low</Badge>
    case "info":
      return <Badge variant="secondary">Info</Badge>
  }
}

const getStatusBadge = (status: string) => {
  switch (status) {
    case "new":
      return <Badge className="bg-blue-500">New</Badge>
    case "acknowledged":
      return (
        <Badge variant="outline" className="border-yellow-500 text-yellow-500">
          Acknowledged
        </Badge>
      )
    case "resolved":
      return (
        <Badge variant="outline" className="border-green-500 text-green-500">
          Resolved
        </Badge>
      )
    default:
      return null
  }
}

export default function AlertsPage() {
  const router = useRouter()
  const [searchQuery, setSearchQuery] = useState("")
  const [severityFilter, setSeverityFilter] = useState<string>("all")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [dateFilter, setDateFilter] = useState<Date | undefined>(undefined)
  const [filteredAlerts, setFilteredAlerts] = useState<Alert[]>(mockAlerts)
  const [sortConfig, setSortConfig] = useState<{
    key: keyof Alert
    direction: "ascending" | "descending"
  }>({
    key: "timestamp",
    direction: "descending",
  })

  // Apply filters
  useEffect(() => {
    let results = [...mockAlerts]

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      results = results.filter(
        (alert) =>
          alert.title.toLowerCase().includes(query) ||
          alert.description.toLowerCase().includes(query) ||
          alert.camera.toLowerCase().includes(query) ||
          alert.location.toLowerCase().includes(query),
      )
    }

    // Severity filter
    if (severityFilter !== "all") {
      results = results.filter((alert) => alert.severity === severityFilter)
    }

    // Status filter
    if (statusFilter !== "all") {
      results = results.filter((alert) => alert.status === statusFilter)
    }

    // Date filter
    if (dateFilter) {
      results = results.filter(
        (alert) =>
          alert.timestamp.getDate() === dateFilter.getDate() &&
          alert.timestamp.getMonth() === dateFilter.getMonth() &&
          alert.timestamp.getFullYear() === dateFilter.getFullYear(),
      )
    }

    // Sort
    results.sort((a, b) => {
      if (sortConfig.key === "timestamp") {
        return sortConfig.direction === "ascending"
          ? a.timestamp.getTime() - b.timestamp.getTime()
          : b.timestamp.getTime() - a.timestamp.getTime()
      } else {
        const aValue = a[sortConfig.key]
        const bValue = b[sortConfig.key]

        if (typeof aValue === "string" && typeof bValue === "string") {
          return sortConfig.direction === "ascending" ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue)
        }

        return 0
      }
    })

    setFilteredAlerts(results)
  }, [searchQuery, severityFilter, statusFilter, dateFilter, sortConfig])

  const handleSort = (key: keyof Alert) => {
    setSortConfig({
      key,
      direction: sortConfig.key === key && sortConfig.direction === "ascending" ? "descending" : "ascending",
    })
  }

  const viewAlertInChat = (alert: Alert) => {
    if (alert.videoId) {
      router.push(`/chat?q=Show+me+${encodeURIComponent(alert.title)}`)
    }
  }

  return (
    <div className="flex flex-col gap-4 sm:gap-6 p-4 sm:p-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Alerts</h1>
        <p className="text-muted-foreground">Monitor and manage system alerts and anomalies</p>
      </div>

      <div className="grid gap-6 md:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Critical Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockAlerts.filter((a) => a.severity === "critical").length}</div>
            <p className="text-xs text-muted-foreground">
              {mockAlerts.filter((a) => a.severity === "critical" && a.status === "new").length} unresolved
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">High Priority</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockAlerts.filter((a) => a.severity === "high").length}</div>
            <p className="text-xs text-muted-foreground">
              {mockAlerts.filter((a) => a.severity === "high" && a.status === "new").length} unresolved
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Medium Priority</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockAlerts.filter((a) => a.severity === "medium").length}</div>
            <p className="text-xs text-muted-foreground">
              {mockAlerts.filter((a) => a.severity === "medium" && a.status === "new").length} unresolved
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Low Priority</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockAlerts.filter((a) => a.severity === "low").length}</div>
            <p className="text-xs text-muted-foreground">
              {mockAlerts.filter((a) => a.severity === "low" && a.status === "new").length} unresolved
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Alert Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <AlertsChart />
        </CardContent>
      </Card>

      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-1 items-center gap-2">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                className="pl-10"
                placeholder="Search alerts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button variant="outline" size="icon">
              <Filter className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Filter by severity" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Severities</SelectItem>
                <SelectItem value="critical">Critical</SelectItem>
                <SelectItem value="high">High</SelectItem>
                <SelectItem value="medium">Medium</SelectItem>
                <SelectItem value="low">Low</SelectItem>
                <SelectItem value="info">Info</SelectItem>
              </SelectContent>
            </Select>

            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="new">New</SelectItem>
                <SelectItem value="acknowledged">Acknowledged</SelectItem>
                <SelectItem value="resolved">Resolved</SelectItem>
              </SelectContent>
            </Select>

            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" className="w-[150px] justify-start">
                  <Calendar className="mr-2 h-4 w-4" />
                  {dateFilter ? format(dateFilter, "PPP") : "Pick a date"}
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-auto p-0">
                <CalendarComponent mode="single" selected={dateFilter} onSelect={setDateFilter} initialFocus />
              </PopoverContent>
            </Popover>

            {dateFilter && (
              <Button variant="ghost" size="sm" onClick={() => setDateFilter(undefined)}>
                Clear Date
              </Button>
            )}
          </div>
        </div>

        <Tabs defaultValue="all">
          <TabsList>
            <TabsTrigger value="all">All Alerts</TabsTrigger>
            <TabsTrigger value="new">New</TabsTrigger>
            <TabsTrigger value="acknowledged">Acknowledged</TabsTrigger>
            <TabsTrigger value="resolved">Resolved</TabsTrigger>
          </TabsList>
          <TabsContent value="all" className="mt-4">
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[180px]">
                        <Button variant="ghost" className="p-0 font-medium" onClick={() => handleSort("title")}>
                          Alert
                          {sortConfig.key === "title" && <ArrowUpDown className="ml-2 h-4 w-4" />}
                        </Button>
                      </TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>
                        <Button variant="ghost" className="p-0 font-medium" onClick={() => handleSort("severity")}>
                          Severity
                          {sortConfig.key === "severity" && <ArrowUpDown className="ml-2 h-4 w-4" />}
                        </Button>
                      </TableHead>
                      <TableHead>
                        <Button variant="ghost" className="p-0 font-medium" onClick={() => handleSort("timestamp")}>
                          Time
                          {sortConfig.key === "timestamp" && <ArrowUpDown className="ml-2 h-4 w-4" />}
                        </Button>
                      </TableHead>
                      <TableHead>
                        <Button variant="ghost" className="p-0 font-medium" onClick={() => handleSort("camera")}>
                          Camera
                          {sortConfig.key === "camera" && <ArrowUpDown className="ml-2 h-4 w-4" />}
                        </Button>
                      </TableHead>
                      <TableHead>
                        <Button variant="ghost" className="p-0 font-medium" onClick={() => handleSort("status")}>
                          Status
                          {sortConfig.key === "status" && <ArrowUpDown className="ml-2 h-4 w-4" />}
                        </Button>
                      </TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredAlerts.map((alert) => (
                      <TableRow key={alert.id}>
                        <TableCell className="font-medium">
                          <div className="flex items-center gap-2">
                            {alert.severity === "critical" || alert.severity === "high" ? (
                              <AlertTriangle className="h-4 w-4 text-red-500" />
                            ) : alert.severity === "medium" ? (
                              <Bell className="h-4 w-4 text-yellow-500" />
                            ) : (
                              <Bell className="h-4 w-4 text-muted-foreground" />
                            )}
                            {alert.title}
                          </div>
                        </TableCell>
                        <TableCell>{alert.description}</TableCell>
                        <TableCell>{getSeverityBadge(alert.severity)}</TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            <Clock className="h-3 w-3 text-muted-foreground" />
                            <span>
                              {alert.timestamp.toLocaleDateString()}{" "}
                              {alert.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                            </span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1">
                            <Camera className="h-3 w-3 text-muted-foreground" />
                            <span>{alert.camera}</span>
                            <span className="text-muted-foreground">({alert.location})</span>
                          </div>
                        </TableCell>
                        <TableCell>{getStatusBadge(alert.status)}</TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            {alert.videoId && (
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => viewAlertInChat(alert)}
                                title="View in Chat"
                              >
                                <Eye className="h-4 w-4" />
                              </Button>
                            )}
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon">
                                  <MoreHorizontal className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuLabel>Actions</DropdownMenuLabel>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem>Acknowledge</DropdownMenuItem>
                                <DropdownMenuItem>Resolve</DropdownMenuItem>
                                <DropdownMenuItem>Assign</DropdownMenuItem>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem>View Details</DropdownMenuItem>
                                {alert.videoId && (
                                  <DropdownMenuItem onClick={() => viewAlertInChat(alert)}>View Video</DropdownMenuItem>
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="new" className="mt-4">
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[180px]">Alert</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>Severity</TableHead>
                      <TableHead>Time</TableHead>
                      <TableHead>Camera</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredAlerts
                      .filter((alert) => alert.status === "new")
                      .map((alert) => (
                        <TableRow key={alert.id}>
                          <TableCell className="font-medium">
                            <div className="flex items-center gap-2">
                              {alert.severity === "critical" || alert.severity === "high" ? (
                                <AlertTriangle className="h-4 w-4 text-red-500" />
                              ) : alert.severity === "medium" ? (
                                <Bell className="h-4 w-4 text-yellow-500" />
                              ) : (
                                <Bell className="h-4 w-4 text-muted-foreground" />
                              )}
                              {alert.title}
                            </div>
                          </TableCell>
                          <TableCell>{alert.description}</TableCell>
                          <TableCell>{getSeverityBadge(alert.severity)}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              <Clock className="h-3 w-3 text-muted-foreground" />
                              <span>
                                {alert.timestamp.toLocaleDateString()}{" "}
                                {alert.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                              </span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              <Camera className="h-3 w-3 text-muted-foreground" />
                              <span>{alert.camera}</span>
                              <span className="text-muted-foreground">({alert.location})</span>
                            </div>
                          </TableCell>
                          <TableCell>{getStatusBadge(alert.status)}</TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-2">
                              {alert.videoId && (
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => viewAlertInChat(alert)}
                                  title="View in Chat"
                                >
                                  <Eye className="h-4 w-4" />
                                </Button>
                              )}
                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <Button variant="ghost" size="icon">
                                    <MoreHorizontal className="h-4 w-4" />
                                  </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                  <DropdownMenuLabel>Actions</DropdownMenuLabel>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem>Acknowledge</DropdownMenuItem>
                                  <DropdownMenuItem>Resolve</DropdownMenuItem>
                                  <DropdownMenuItem>Assign</DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem>View Details</DropdownMenuItem>
                                  {alert.videoId && (
                                    <DropdownMenuItem onClick={() => viewAlertInChat(alert)}>
                                      View Video
                                    </DropdownMenuItem>
                                  )}
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="acknowledged" className="mt-4">
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[180px]">Alert</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>Severity</TableHead>
                      <TableHead>Time</TableHead>
                      <TableHead>Camera</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredAlerts
                      .filter((alert) => alert.status === "acknowledged")
                      .map((alert) => (
                        <TableRow key={alert.id}>
                          <TableCell className="font-medium">
                            <div className="flex items-center gap-2">
                              {alert.severity === "critical" || alert.severity === "high" ? (
                                <AlertTriangle className="h-4 w-4 text-red-500" />
                              ) : alert.severity === "medium" ? (
                                <Bell className="h-4 w-4 text-yellow-500" />
                              ) : (
                                <Bell className="h-4 w-4 text-muted-foreground" />
                              )}
                              {alert.title}
                            </div>
                          </TableCell>
                          <TableCell>{alert.description}</TableCell>
                          <TableCell>{getSeverityBadge(alert.severity)}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              <Clock className="h-3 w-3 text-muted-foreground" />
                              <span>
                                {alert.timestamp.toLocaleDateString()}{" "}
                                {alert.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                              </span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              <Camera className="h-3 w-3 text-muted-foreground" />
                              <span>{alert.camera}</span>
                              <span className="text-muted-foreground">({alert.location})</span>
                            </div>
                          </TableCell>
                          <TableCell>{getStatusBadge(alert.status)}</TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-2">
                              {alert.videoId && (
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => viewAlertInChat(alert)}
                                  title="View in Chat"
                                >
                                  <Eye className="h-4 w-4" />
                                </Button>
                              )}
                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <Button variant="ghost" size="icon">
                                    <MoreHorizontal className="h-4 w-4" />
                                  </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                  <DropdownMenuLabel>Actions</DropdownMenuLabel>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem>Resolve</DropdownMenuItem>
                                  <DropdownMenuItem>Assign</DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem>View Details</DropdownMenuItem>
                                  {alert.videoId && (
                                    <DropdownMenuItem onClick={() => viewAlertInChat(alert)}>
                                      View Video
                                    </DropdownMenuItem>
                                  )}
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="resolved" className="mt-4">
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[180px]">Alert</TableHead>
                      <TableHead>Description</TableHead>
                      <TableHead>Severity</TableHead>
                      <TableHead>Time</TableHead>
                      <TableHead>Camera</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredAlerts
                      .filter((alert) => alert.status === "resolved")
                      .map((alert) => (
                        <TableRow key={alert.id}>
                          <TableCell className="font-medium">
                            <div className="flex items-center gap-2">
                              {alert.severity === "critical" || alert.severity === "high" ? (
                                <AlertTriangle className="h-4 w-4 text-red-500" />
                              ) : alert.severity === "medium" ? (
                                <Bell className="h-4 w-4 text-yellow-500" />
                              ) : (
                                <Bell className="h-4 w-4 text-muted-foreground" />
                              )}
                              {alert.title}
                            </div>
                          </TableCell>
                          <TableCell>{alert.description}</TableCell>
                          <TableCell>{getSeverityBadge(alert.severity)}</TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              <Clock className="h-3 w-3 text-muted-foreground" />
                              <span>
                                {alert.timestamp.toLocaleDateString()}{" "}
                                {alert.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                              </span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              <Camera className="h-3 w-3 text-muted-foreground" />
                              <span>{alert.camera}</span>
                              <span className="text-muted-foreground">({alert.location})</span>
                            </div>
                          </TableCell>
                          <TableCell>{getStatusBadge(alert.status)}</TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-2">
                              {alert.videoId && (
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => viewAlertInChat(alert)}
                                  title="View in Chat"
                                >
                                  <Eye className="h-4 w-4" />
                                </Button>
                              )}
                              <DropdownMenu>
                                <DropdownMenuTrigger asChild>
                                  <Button variant="ghost" size="icon">
                                    <MoreHorizontal className="h-4 w-4" />
                                  </Button>
                                </DropdownMenuTrigger>
                                <DropdownMenuContent align="end">
                                  <DropdownMenuLabel>Actions</DropdownMenuLabel>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem>Reopen</DropdownMenuItem>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem>View Details</DropdownMenuItem>
                                  {alert.videoId && (
                                    <DropdownMenuItem onClick={() => viewAlertInChat(alert)}>
                                      View Video
                                    </DropdownMenuItem>
                                  )}
                                </DropdownMenuContent>
                              </DropdownMenu>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
