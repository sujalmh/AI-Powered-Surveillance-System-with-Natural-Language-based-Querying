"use client"

import type React from "react"

import { useState } from "react"
import { useEffect } from "react"
import { useRouter } from "next/navigation"
import {
  ArrowRight,
  ArrowUpRight,
  Bell,
  Car,
  ChevronRight,
  Clock,
  Package,
  Search,
  User,
  Users,
  AlertTriangle,
  Mic,
} from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AnalyticsChart } from "@/components/analytics-chart"
import { ActivityFeed } from "@/components/activity-feed"
import { CameraPreview } from "@/components/camera-preview"

export default function Dashboard() {
  const router = useRouter()
  const [query, setQuery] = useState("")

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) {
      router.push(`/chat?q=${encodeURIComponent(query)}`)
    }
  }
  const [todayCount, setTodayCount] = useState<number | null>(null);
  const [percentChange, setPercentChange] = useState<number | null>(null);

  useEffect(() => {
    fetch('http://localhost:5000/analytics/person-count-delta')
      .then(res => res.json())
      .then(data => {
        setTodayCount(data.today_count);
        setPercentChange(data.percent_change);
      })
      .catch(err => console.error('Error fetching visitor stats:', err));
  }, []);


// vehicles detected
  const [data, setData] = useState({
    today_count: 0,
    percent_change: 0
  });

  useEffect(() => {
    fetch('http://localhost:5000/analytics/vehicle-count-delta')
      .then((res) => res.json())
      .then((json) => setData(json))
      .catch((err) => console.error("Error fetching vehicle data:", err));
  }, []);
  const progressPercent = Math.min(Math.max(data.percent_change, 0), 100);
  // vehicle detected ends

  // Objects Detected
  const [objectStats, setObjectStats] = useState({
    today_count: 0,
    percent_change: 0
  });

  useEffect(() => {
    fetch('http://localhost:5000/analytics/object-count-delta')
      .then((res) => res.json())
      .then((json) => setObjectStats(json))
      .catch((err) => console.error("Error fetching object data:", err));
  }, []);

  const progressPercentage = Math.min(Math.max(objectStats.percent_change, 0), 100); // clamp between 0–100
  const changeSymbol = objectStats.percent_change >= 0 ? "+" : "";
  //Object detected ends here

// People Detected
  const [personStats, setPersonStats] = useState({
    this_week_count: 0,
    percent_change: 0
  });

  useEffect(() => {
    fetch('http://localhost:5000/analytics/person-weekly-delta')
      .then((res) => res.json())
      .then((json) => setPersonStats(json))
      .catch((err) => console.error("Error fetching people stats:", err));
  }, []);

  const progressPercent2 = Math.min(Math.max(personStats.percent_change, 0), 100); // Clamp 0–100
  const changeSymbol2 = personStats.percent_change >= 0 ? "+" : "";
   //People detected ends here

  return (
    <div className="flex flex-col gap-4 sm:gap-6 p-4 sm:p-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">AI-powered surveillance system overview and analytics</p>
      </div>

      <form
        onSubmit={handleSearch}
        className="mx-auto flex w-full max-w-3xl items-center gap-2 animate-in fade-in-50 duration-500"
      >
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            className="pl-10 py-6 text-lg"
            placeholder="Ask about any event or footage..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <Button type="submit" size="lg" className="gap-2">
          Search <ArrowRight className="h-4 w-4" />
        </Button>
        <Button type="button" variant="outline" size="lg" className="px-3">
          <Mic className="h-4 w-4" />
        </Button>
      </form>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card className="hover:shadow-md transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Visitors Today</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{todayCount !== null ? todayCount : '...'}</div>
            <p className="text-xs text-muted-foreground">{percentChange !== null
            ? `${percentChange > 0 ? '+' : ''}${percentChange}% from yesterday`
            : 'Loading...'}</p>
            <div className="mt-4 h-1 w-full rounded-full bg-muted">
              <div className="h-1 w-[75%] rounded-full bg-primary" style={{ width: `${Math.min(Math.abs(percentChange || 0), 100)}%` }}></div>
            </div>
          </CardContent>
        </Card>
        <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">People Detected</CardTitle>
        <User className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">
          {personStats.this_week_count}
        </div>
        <p className="text-xs text-muted-foreground">
          {changeSymbol2}{personStats.percent_change}% from last week
        </p>
        <div className="mt-4 h-1 w-full rounded-full bg-muted">
          <div
            className="h-1 rounded-full bg-primary"
            style={{ width: `${progressPercent2}%` }}
          ></div>
        </div>
      </CardContent>
    </Card>
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">Vehicles Detected</CardTitle>
        <Car className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{data.today_count}</div>
        <p className="text-xs text-muted-foreground">
          {data.percent_change}% from yesterday
        </p>
        <div className="mt-4 h-1 w-full rounded-full bg-muted">
          <div
            className="h-1 rounded-full bg-primary"
            style={{ width: `${progressPercent}%` }}
          ></div>
        </div>
      </CardContent>
    </Card>
        <Card className="hover:shadow-md transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-sm font-medium">Objects Detected</CardTitle>
            <Package className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{objectStats.today_count}</div>
            <p className="text-xs text-muted-foreground">{changeSymbol}{objectStats.percent_change}% from yesterday</p>
            <div className="mt-4 h-1 w-full rounded-full bg-muted">
              <div className="h-1 w-[80%] rounded-full bg-primary"
              style={{ width: `${progressPercentage}%` }}
              ></div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-7">
        <Card className="md:col-span-4 hover:shadow-md transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Activity Trends</CardTitle>
            <Tabs defaultValue="24h">
              <TabsList>
                <TabsTrigger value="24h">24h</TabsTrigger>
                <TabsTrigger value="7d">7d</TabsTrigger>
                <TabsTrigger value="30d">30d</TabsTrigger>
              </TabsList>
            </Tabs>
          </CardHeader>
          <CardContent>
            <AnalyticsChart />
          </CardContent>
        </Card>
        <Card className="md:col-span-3 hover:shadow-md transition-shadow">
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent>
            <ActivityFeed />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card className="hover:shadow-md transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Anomaly Alerts</CardTitle>
            <Button
              variant="outline"
              size="sm"
              className="gap-1"
              onClick={() => router.push("/chat?q=Show%20me%20all%20anomalies")}
            >
              View All <ChevronRight className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className="flex items-start gap-4 rounded-lg border p-3 hover:bg-muted/50 transition-colors cursor-pointer"
              onClick={() => router.push("/chat?q=Show%20unusual%20movement%20in%20storage%20area")}
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-red-100 text-red-700 dark:bg-red-900/20">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold">Unusual Movement Detected</h4>
                  <Badge variant="destructive" className="ml-2">
                    High
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">Camera 3 - Parking Lot</p>
                <div className="mt-1 flex items-center text-xs text-muted-foreground">
                  <Clock className="mr-1 h-3 w-3" />
                  <span>Today, 14:32</span>
                </div>
              </div>
            </div>
            <div
              className="flex items-start gap-4 rounded-lg border p-3 hover:bg-muted/50 transition-colors cursor-pointer"
              onClick={() => router.push("/chat?q=Show%20after%20hours%20access%20at%20main%20entrance")}
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-amber-100 text-amber-700 dark:bg-amber-900/20">
                <Bell className="h-5 w-5" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold">After Hours Access</h4>
                  <Badge variant="outline" className="ml-2">
                    Medium
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">Camera 1 - Main Entrance</p>
                <div className="mt-1 flex items-center text-xs text-muted-foreground">
                  <Clock className="mr-1 h-3 w-3" />
                  <span>Today, 23:15</span>
                </div>
              </div>
            </div>
            <div
              className="flex items-start gap-4 rounded-lg border p-3 hover:bg-muted/50 transition-colors cursor-pointer"
              onClick={() => router.push("/chat?q=Show%20unauthorized%20access%20at%20server%20room")}
            >
              <div className="flex h-9 w-9 items-center justify-center rounded-full bg-red-100 text-red-700 dark:bg-red-900/20">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between">
                  <h4 className="font-semibold">Unauthorized Access Attempt</h4>
                  <Badge variant="destructive" className="ml-2">
                    High
                  </Badge>
                </div>
                <p className="text-sm text-muted-foreground">Camera 10 - Server Room</p>
                <div className="mt-1 flex items-center text-xs text-muted-foreground">
                  <Clock className="mr-1 h-3 w-3" />
                  <span>Today, 02:47</span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="hover:shadow-md transition-shadow">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Live Highlights</CardTitle>
            <Button variant="outline" size="sm" className="gap-1" onClick={() => router.push("/cameras")}>
              All Cameras <ArrowUpRight className="h-4 w-4" />
            </Button>
          </CardHeader>
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <div
              className="cursor-pointer transition-transform hover:scale-[1.02]"
              onClick={() => router.push("/cameras")}
            >
              <CameraPreview name="Main Entrance" location="Front Door" active={true} />
            </div>
            <div
              className="cursor-pointer transition-transform hover:scale-[1.02]"
              onClick={() => router.push("/cameras")}
            >
              <CameraPreview name="Parking Lot" location="North Side" active={true} hasAlert={true} />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="mt-4 flex justify-center gap-4">
        <Button variant="outline" size="lg" className="gap-2" onClick={() => router.push("/chat")}>
          Open AI Chat <ArrowUpRight className="h-4 w-4" />
        </Button>
        <Button variant="outline" size="lg" className="gap-2" onClick={() => router.push("/cameras")}>
          View All Cameras <ArrowUpRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
