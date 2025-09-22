"use client"

import type React from "react"

import { useState } from "react"
import { Plus, Upload, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { DialogFooter } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"

// Sample locations - in a real app, these would come from your backend
const existingLocations = [
  "Front Door",
  "Back Door",
  "Parking Lot",
  "Reception Area",
  "Warehouse",
  "Server Room",
  "Lobby",
  "Conference Room",
  "Hallway",
  "Office Area",
]

export function AddCameraForm() {
  const [cameraName, setCameraName] = useState("")
  const [selectedLocation, setSelectedLocation] = useState("")
  const [newLocation, setNewLocation] = useState("")
  const [isAddingLocation, setIsAddingLocation] = useState(false)
  const [resolution, setResolution] = useState("1080p")
  const [fps, setFps] = useState("30")
  const [enableMotionDetection, setEnableMotionDetection] = useState(true)
  const [enableAudio, setEnableAudio] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Check if the file is a video
    if (!file.type.startsWith("video/")) {
      alert("Please select a video file")
      return
    }

    setSelectedFile(file)

    // Create a preview URL for the video
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  // Handle adding a new location
  const handleAddLocation = () => {
    if (!newLocation.trim()) return
    setSelectedLocation(newLocation)
    setNewLocation("")
    setIsAddingLocation(false)
  }

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
  
    const formData = new FormData()
    formData.append("name", cameraName)
    formData.append("location", selectedLocation)
    formData.append("resolution", resolution)
    formData.append("fps", fps)
    formData.append("status", enableMotionDetection ? "true" : "false")
    formData.append("audio", enableAudio ? "true" : "false")
  
    if (selectedFile) {
      formData.append("file", selectedFile)
    }
  
    try {
      const response = await fetch("http://localhost:5000/cam/add/camera", {
        method: "POST",
        body: formData,
      })
  
      if (response.ok) {
        const data = await response.json()
        console.log("Camera added:", data)
  
        // Reset form
        setCameraName("")
        setSelectedLocation("")
        setResolution("1080p")
        setFps("30")
        setEnableMotionDetection(true)
        setEnableAudio(false)
        setSelectedFile(null)
        setPreviewUrl(null)
  
        // Optional: Show a toast/success modal
        alert("Camera added successfully!")
      } else {
        const err = await response.json()
        console.error("Failed to add camera:", err)
        // Show error toast or message
      }
    } catch (error) {
      console.error("Error during camera add:", error)
    }
  }
  

  // Clean up preview URL when component unmounts or when file changes
  const clearPreview = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
      setPreviewUrl(null)
      setSelectedFile(null)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="grid gap-6 py-4">
        <div className="space-y-2">
          <Label htmlFor="camera-name">Camera Name</Label>
          <Input
            id="camera-name"
            placeholder="Enter camera name"
            value={cameraName}
            onChange={(e) => setCameraName(e.target.value)}
            required
          />
        </div>

        <div className="space-y-2">
          <Label htmlFor="camera-location">Location</Label>
          <div className="flex gap-2">
            <Select value={selectedLocation} onValueChange={setSelectedLocation} required>
              <SelectTrigger id="camera-location" className="flex-1">
                <SelectValue placeholder="Select location" />
              </SelectTrigger>
              <SelectContent>
                {existingLocations.map((location) => (
                  <SelectItem key={location} value={location}>
                    {location}
                  </SelectItem>
                ))}
                <SelectItem value="add-new" className="text-primary">
                  + Add New Location
                </SelectItem>
              </SelectContent>
            </Select>

            {selectedLocation === "add-new" && (
              <Popover open={isAddingLocation} onOpenChange={setIsAddingLocation}>
                <PopoverTrigger asChild>
                  <Button variant="outline" onClick={() => setIsAddingLocation(true)}>
                    <Plus className="h-4 w-4 mr-2" /> New Location
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-80">
                  <div className="space-y-4">
                    <h4 className="font-medium">Add New Location</h4>
                    <div className="space-y-2">
                      <Input
                        placeholder="Enter location name"
                        value={newLocation}
                        onChange={(e) => setNewLocation(e.target.value)}
                      />
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button variant="outline" size="sm" onClick={() => setIsAddingLocation(false)}>
                        Cancel
                      </Button>
                      <Button size="sm" onClick={handleAddLocation} disabled={!newLocation.trim()}>
                        Add
                      </Button>
                    </div>
                  </div>
                </PopoverContent>
              </Popover>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="resolution">Resolution</Label>
            <Select value={resolution} onValueChange={setResolution}>
              <SelectTrigger id="resolution">
                <SelectValue placeholder="Select resolution" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="720p">720p</SelectItem>
                <SelectItem value="1080p">1080p</SelectItem>
                <SelectItem value="4K">4K</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="fps">Frame Rate</Label>
            <Select value={fps} onValueChange={setFps}>
              <SelectTrigger id="fps">
                <SelectValue placeholder="Select FPS" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="15">15 FPS</SelectItem>
                <SelectItem value="30">30 FPS</SelectItem>
                <SelectItem value="60">60 FPS</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <div className="space-y-2">
          <Label>Camera Feed</Label>
          <div className="border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center">
            {previewUrl ? (
              <div className="space-y-4 w-full">
                <div className="relative aspect-video w-full bg-black rounded-md overflow-hidden">
                  <video src={previewUrl} className="w-full h-full object-contain" controls />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 text-white rounded-full"
                    onClick={clearPreview}
                    type="button"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                <div className="flex items-center">
                  <Badge variant="outline" className="text-xs">
                    {selectedFile?.name}
                  </Badge>
                </div>
              </div>
            ) : (
              <div className="space-y-4 text-center">
                <div className="mx-auto bg-muted rounded-full p-3">
                  <Upload className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">Upload camera feed</p>
                  <p className="text-xs text-muted-foreground">Drag and drop a video file or click to browse</p>
                </div>
                <Input id="camera-feed" type="file" accept="video/*" className="hidden" onChange={handleFileChange} />
                <Button variant="outline" onClick={() => document.getElementById("camera-feed")?.click()} type="button">
                  Select Video
                </Button>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="motion-detection">Status of the camera</Label>
              <p className="text-xs text-muted-foreground">Status of the camera "online" or "offline"</p> 
            </div>
            <Switch id="motion-detection" checked={enableMotionDetection} onCheckedChange={setEnableMotionDetection} />
          </div>
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <Label htmlFor="audio-recording">Audio Recording</Label>
              <p className="text-xs text-muted-foreground">Record audio with video footage</p>
            </div>
            <Switch id="audio-recording" checked={enableAudio} onCheckedChange={setEnableAudio} />
          </div>
        </div>
      </div>
      <DialogFooter>
        <Button variant="outline" type="button">
          Cancel
        </Button>
        <Button type="submit">Add Camera</Button>
      </DialogFooter>
    </form>
  )
}
