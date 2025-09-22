"use client"

import type React from "react"
import { useState, useEffect } from "react"
import { Search, Grid3X3, Grid2X2, Mic, ArrowUpRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { CameraPreview } from "@/components/camera-preview"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { VideoModal } from "@/components/video-modal"
import { useRouter } from "next/navigation"

type Camera = {
  id: string
  name: string
  location: string
  type: "indoor" | "outdoor"
  zone: string
  hasAlerts: boolean
  active: boolean
}

export default function CamerasPage() {
  const router = useRouter()
  const [searchQuery, setSearchQuery] = useState("")
  const [nlQuery, setNlQuery] = useState("")
  const [gridSize, setGridSize] = useState<"2x2" | "3x3" | "4x4">("3x3")
  const [statusFilter, setStatusFilter] = useState<"all" | "active" | "inactive">("all")
  const [zoneFilter, setZoneFilter] = useState<string>("all")
  const [selectedCamera, setSelectedCamera] = useState<Camera | null>(null)
  const [isVideoModalOpen, setIsVideoModalOpen] = useState(false)
  const [cameras, setCameras] = useState<Camera[]>([])
  const [filteredCameras, setFilteredCameras] = useState<Camera[]>([])

  // Fetch cameras from API
  const fetchCameras = async () => {
    try {
      const response = await fetch("http://localhost:5000/cam/list/cameras")
      if (!response.ok) {
        throw new Error("Failed to fetch cameras")
      }
      const data: Camera[] = await response.json()
      setCameras(data)
      setFilteredCameras(data)
    } catch (error) {
      console.error("Error fetching cameras:", error)
    }
  }

  useEffect(() => {
    fetchCameras()
  }, [])

  // Process natural language query
  useEffect(() => {
    if (!nlQuery.trim()) {
      applyFilters()
      return
    }

    const query = nlQuery.toLowerCase()
    let results = [...cameras]

    // Handle location-based queries
    if (query.includes("entrance") || query.includes("door")) {
      results = results.filter(
        (cam) =>
          cam.name.toLowerCase().includes("entrance") ||
          cam.name.toLowerCase().includes("door") ||
          cam.zone.toLowerCase() === "entrance",
      )
    }

    if (query.includes("parking") || query.includes("garage")) {
      results = results.filter(
        (cam) =>
          cam.name.toLowerCase().includes("parking") ||
          cam.name.toLowerCase().includes("garage") ||
          cam.zone.toLowerCase() === "parking",
      )
    }

    if (query.includes("office") || query.includes("executive")) {
      results = results.filter(
        (cam) => cam.name.toLowerCase().includes("office") || cam.zone.toLowerCase() === "office",
      )
    }

    // Handle type-based queries
    if (query.includes("indoor") || query.includes("inside")) {
      results = results.filter((cam) => cam.type === "indoor")
    }

    if (query.includes("outdoor") || query.includes("outside")) {
      results = results.filter((cam) => cam.type === "outdoor")
    }

    // Handle status-based queries
    if (query.includes("active") || query.includes("online")) {
      results = results.filter((cam) => cam.active)
    }

    if (query.includes("inactive") || query.includes("offline")) {
      results = results.filter((cam) => !cam.active)
    }

    // Handle alert-based queries
    if (query.includes("alert") || query.includes("warning")) {
      results = results.filter((cam) => cam.hasAlerts)
    }

    setFilteredCameras(results)
  }, [nlQuery, cameras])

  // Apply regular filters
  const applyFilters = () => {
    let results = [...cameras]

    // Apply search filter
    if (searchQuery) {
      results = results.filter(
        (camera) =>
          camera.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          camera.location.toLowerCase().includes(searchQuery.toLowerCase()) ||
          camera.zone.toLowerCase().includes(searchQuery.toLowerCase()),
      )
    }

    // Apply status filter
    if (statusFilter !== "all") {
      results = results.filter((camera) => (statusFilter === "active" ? camera.active : !camera.active))
    }

    // Apply zone filter
    if (zoneFilter !== "all") {
      results = results.filter((camera) => camera.zone === zoneFilter)
    }

    setFilteredCameras(results)
  }

  useEffect(() => {
    applyFilters()
  }, [searchQuery, statusFilter, zoneFilter, cameras])

  const handleNaturalLanguageSearch = (e: React.FormEvent) => {
    e.preventDefault()
    // The effect will handle the filtering
  }

  const openVideoModal = async (camera: Camera) => {
    try {
      const response = await fetch(`http://localhost:5000/cam/stream/${camera.id}`)
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Failed to initiate camera stream")
      }
      setSelectedCamera(camera)
      setIsVideoModalOpen(true)
    } catch (error) {
      console.error("Error initiating camera stream:", error)
    }
  }

  const zones = Array.from(new Set(cameras.map((cam) => cam.zone)))

  return (
    <div className="flex flex-col gap-4 sm:gap-6 p-4 sm:p-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Camera Feeds</h1>
        <p className="text-muted-foreground">View and search all surveillance cameras</p>
      </div>

      <form onSubmit={handleNaturalLanguageSearch} className="flex flex-col gap-4 md:flex-row md:items-center">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            className="pl-10 py-6"
            placeholder="Search using natural language (e.g., 'Show me all entrance cameras')"
            value={nlQuery}
            onChange={(e) => setNlQuery(e.target.value)}
          />
        </div>
        <div className="flex gap-2">
          <Button type="submit" className="shrink-0">
            Search
          </Button>
          <Button type="button" variant="outline" size="icon" className="shrink-0" title="Voice search">
            <Mic className="h-4 w-4" />
          </Button>
        </div>
      </form>

      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <Select value={statusFilter} onValueChange={(value: "all" | "active" | "inactive") => setStatusFilter(value)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Filter by status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Cameras</SelectItem>
              <SelectItem value="active">Active Only</SelectItem>
              <SelectItem value="inactive">Inactive Only</SelectItem>
            </SelectContent>
          </Select>

          <Select value={zoneFilter} onValueChange={setZoneFilter}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder="Filter by zone" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Zones</SelectItem>
              {zones.map((zone) => (
                <SelectItem key={zone} value={zone}>
                  {zone}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div className="flex items-center rounded-md border">
            <Button
              variant="ghost"
              size="icon"
              className={`rounded-none rounded-l-md ${gridSize === "2x2" ? "bg-muted" : ""}`}
              onClick={() => setGridSize("2x2")}
            >
              <Grid2X2 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={`rounded-none ${gridSize === "3x3" ? "bg-muted" : ""}`}
              onClick={() => setGridSize("3x3")}
            >
              <Grid3X3 className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className={`rounded-none rounded-r-md ${gridSize === "4x4" ? "bg-muted" : ""}`}
              onClick={() => setGridSize("4x4")}
            >
              <Grid3X3 className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div>
          <Button variant="outline" size="sm" className="gap-1" onClick={() => router.push("/chat")}>
            AI Chat <ArrowUpRight className="h-3 w-3" />
          </Button>
        </div>
      </div>

      <Tabs defaultValue="all">
        <TabsList>
          <TabsTrigger value="all">All Cameras ({cameras.length})</TabsTrigger>
          <TabsTrigger value="indoor">Indoor ({cameras.filter((c) => c.type === "indoor").length})</TabsTrigger>
          <TabsTrigger value="outdoor">Outdoor ({cameras.filter((c) => c.type === "outdoor").length})</TabsTrigger>
          <TabsTrigger value="alerts">With Alerts ({cameras.filter((c) => c.hasAlerts).length})</TabsTrigger>
        </TabsList>
        <TabsContent value="all" className="mt-4">
          <div
            className={`camera-grid ${
              gridSize === "2x2"
                ? "grid-cols-1 md:grid-cols-2"
                : gridSize === "3x3"
                  ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
                  : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
            }`}
          >
            {filteredCameras.map((camera) => (
              <div
                key={camera.id}
                onClick={() => openVideoModal(camera)}
                className="transition-transform hover:scale-[1.02] cursor-pointer"
              >
                <CameraPreview
                  id = {camera.id}
                  name={camera.name}
                  location={camera.location}
                  active={camera.active}
                  hasAlert={camera.hasAlerts}
                />
              </div>
            ))}
          </div>
        </TabsContent>
        <TabsContent value="indoor" className="mt-4">
          <div
            className={`camera-grid ${
              gridSize === "2x2"
                ? "grid-cols-1 md:grid-cols-2"
                : gridSize === "3x3"
                  ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
                  : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
            }`}
          >
            {filteredCameras
              .filter((camera) => camera.type === "indoor")
              .map((camera) => (
                <div
                  key={camera.id}
                  onClick={() => openVideoModal(camera)}
                  className="transition-transform hover:scale-[1.02] cursor-pointer"
                >
                  <CameraPreview
                    id = {camera.id}
                    name={camera.name}
                    location={camera.location}
                    active={camera.active}
                    hasAlert={camera.hasAlerts}
                  />
                </div>
              ))}
          </div>
        </TabsContent>
        <TabsContent value="outdoor" className="mt-4">
          <div
            className={`camera-grid ${
              gridSize === "2x2"
                ? "grid-cols-1 md:grid-cols-2"
                : gridSize === "3x3"
                  ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
                  : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
            }`}
          >
            {filteredCameras
              .filter((camera) => camera.type === "outdoor")
              .map((camera) => (
                <div
                  key={camera.id}
                  onClick={() => openVideoModal(camera)}
                  className="transition-transform hover:scale-[1.02] cursor-pointer"
                >
                  <CameraPreview
                    id = {camera.id}
                    name={camera.name}
                    location={camera.location}
                    active={camera.active}
                    hasAlert={camera.hasAlerts}
                  />
                </div>
              ))}
          </div>
        </TabsContent>
        <TabsContent value="alerts" className="mt-4">
          <div
            className={`camera-grid ${
              gridSize === "2x2"
                ? "grid-cols-1 md:grid-cols-2"
                : gridSize === "3x3"
                  ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3"
                  : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4"
            }`}
          >
            {filteredCameras
              .filter((camera) => camera.hasAlerts)
              .map((camera) => (
                <div
                  key={camera.id}
                  onClick={() => openVideoModal(camera)}
                  className="transition-transform hover:scale-[1.02] cursor-pointer"
                >
                  <CameraPreview
                    id = {camera.id}
                    name={camera.name}
                    location={camera.location}
                    active={camera.active}
                    hasAlert={camera.hasAlerts}
                  />
                </div>
              ))}
          </div>
        </TabsContent>
      </Tabs>

      {selectedCamera && (
        <VideoModal
        isOpen={isVideoModalOpen}
        onClose={() => setIsVideoModalOpen(false)}
        onActivate={fetchCameras} // Added to refresh camera list
        video={{
          id: selectedCamera.id,
          title: selectedCamera.name,
          timestamp: new Date().toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
          location: selectedCamera.location,
          active: selectedCamera.active, // Added
        }}
      />
      )}
    </div>
  )
}