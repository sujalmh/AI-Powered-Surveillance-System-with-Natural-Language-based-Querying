"use client"

import { useEffect,useState } from "react"
import { Save, Camera, Trash2, Info, AlertTriangle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { StorageChart } from "@/components/storage-chart"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { AddCameraForm } from "@/components/add-camera-form"

// type CameraType = {
//   id: string
//   name: string
//   location: string
//   resolution: string
//   fps: number
//   status:boolean
//   storage: number
// }
interface Camera {
  id: string
  name: string
  location: string
  resolution: string
  fps: number
  status: boolean
  storage: number
}



// const mockCameras: CameraType[] = [
//   {
//     id: "cam1",
//     name: "Main Entrance",
//     location: "Front Door",
//     resolution: "1080p",
//     fps: 30,
//     status: "online",
//     storage: 15.2,
//   },
//   {
//     id: "cam2",
//     name: "Parking Lot",
//     location: "North Side",
//     resolution: "1080p",
//     fps: 15,
//     status: "online",
//     storage: 12.8,
//   },
//   {
//     id: "cam3",
//     name: "Lobby",
//     location: "Reception Area",
//     resolution: "4K",
//     fps: 30,
//     status: "online",
//     storage: 28.5,
//   },
//   {
//     id: "cam4",
//     name: "Storage Room",
//     location: "Basement",
//     resolution: "720p",
//     fps: 15,
//     status: "online",
//     storage: 8.7,
//   },
//   {
//     id: "cam5",
//     name: "Conference Room",
//     location: "2nd Floor",
//     resolution: "1080p",
//     fps: 30,
//     status: "online",
//     storage: 14.3,
//   },
//   {
//     id: "cam6",
//     name: "Side Entrance",
//     location: "East Wing",
//     resolution: "1080p",
//     fps: 15,
//     status: "offline",
//     storage: 0,
//   },
// ]

export default function SettingsPage() {
  const [timeZone, setTimeZone] = useState("America/New_York")
  const [dateFormat, setDateFormat] = useState("MM/DD/YYYY")
  const [timeFormat, setTimeFormat] = useState("12h")
  const [language, setLanguage] = useState("en-US")
  const [retentionDays, setRetentionDays] = useState<number[]>([30])
  const [totalStorage, setTotalStorage] = useState(512)
  const [usedStorage, setUsedStorage] = useState(0)
  const [cameraList, setCameraList] = useState<Camera[]>([])

  const handleRetentionChange = (value: number[]) => {
    setRetentionDays(value)
  }


  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const res = await fetch("http://localhost:5000/cam/list/cameras")
        const data = await res.json()
        setCameraList(data)
        console.log("Camera List", data)
        const totalUsedStorage = data.reduce((acc: number, camera: Camera) => acc + parseFloat(camera.storage), 0)
        setUsedStorage(totalUsedStorage)
        console.log("Used Storage", totalUsedStorage)
      } catch (err) {
        console.error("Error fetching cameras:", err)
      }
    }
  
    fetchCameras()
  }, [])


  return (
    <div className="flex flex-col gap-4 sm:gap-6 p-4 sm:p-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">Configure system settings and preferences</p>
      </div>

      <Tabs defaultValue="general">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="cameras">Cameras</TabsTrigger>
          <TabsTrigger value="storage">Storage</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
        </TabsList>
        <TabsContent value="general" className="mt-6 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Regional Settings</CardTitle>
                <CardDescription>Configure time zone, date and time formats</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="timezone">Time Zone</Label>
                  <Select value={timeZone} onValueChange={setTimeZone}>
                    <SelectTrigger id="timezone">
                      <SelectValue placeholder="Select time zone" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="America/New_York">Eastern Time (ET)</SelectItem>
                      <SelectItem value="America/Chicago">Central Time (CT)</SelectItem>
                      <SelectItem value="America/Denver">Mountain Time (MT)</SelectItem>
                      <SelectItem value="America/Los_Angeles">Pacific Time (PT)</SelectItem>
                      <SelectItem value="Europe/London">London (GMT)</SelectItem>
                      <SelectItem value="Europe/Paris">Paris (CET)</SelectItem>
                      <SelectItem value="Asia/Tokyo">Tokyo (JST)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="date-format">Date Format</Label>
                  <Select value={dateFormat} onValueChange={setDateFormat}>
                    <SelectTrigger id="date-format">
                      <SelectValue placeholder="Select date format" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="MM/DD/YYYY">MM/DD/YYYY</SelectItem>
                      <SelectItem value="DD/MM/YYYY">DD/MM/YYYY</SelectItem>
                      <SelectItem value="YYYY-MM-DD">YYYY-MM-DD</SelectItem>
                      <SelectItem value="MMM D, YYYY">MMM D, YYYY</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="time-format">Time Format</Label>
                  <Select value={timeFormat} onValueChange={setTimeFormat}>
                    <SelectTrigger id="time-format">
                      <SelectValue placeholder="Select time format" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="12h">12-hour (AM/PM)</SelectItem>
                      <SelectItem value="24h">24-hour</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="language">Language</Label>
                  <Select value={language} onValueChange={setLanguage}>
                    <SelectTrigger id="language">
                      <SelectValue placeholder="Select language" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en-US">English (US)</SelectItem>
                      <SelectItem value="en-GB">English (UK)</SelectItem>
                      <SelectItem value="es">Spanish</SelectItem>
                      <SelectItem value="fr">French</SelectItem>
                      <SelectItem value="de">German</SelectItem>
                      <SelectItem value="ja">Japanese</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline">Reset to Defaults</Button>
                <Button>Save Changes</Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>System Settings</CardTitle>
                <CardDescription>Configure general system behavior</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="auto-logout">Auto Logout</Label>
                      <p className="text-xs text-muted-foreground">Automatically log out inactive users</p>
                    </div>
                    <Switch id="auto-logout" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="inactivity-timeout">Inactivity Timeout</Label>
                    <Select defaultValue="30">
                      <SelectTrigger id="inactivity-timeout" className="w-[180px]">
                        <SelectValue placeholder="Select timeout" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="15">15 minutes</SelectItem>
                        <SelectItem value="30">30 minutes</SelectItem>
                        <SelectItem value="60">1 hour</SelectItem>
                        <SelectItem value="120">2 hours</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <Separator />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="dark-mode">Dark Mode</Label>
                      <p className="text-xs text-muted-foreground">Use dark theme for the interface</p>
                    </div>
                    <Switch id="dark-mode" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="animations">Interface Animations</Label>
                      <p className="text-xs text-muted-foreground">Enable smooth animations in the UI</p>
                    </div>
                    <Switch id="animations" defaultChecked />
                  </div>
                </div>
                <Separator />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="analytics">Usage Analytics</Label>
                      <p className="text-xs text-muted-foreground">Collect anonymous usage data</p>
                    </div>
                    <Switch id="analytics" />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="crash-reports">Crash Reports</Label>
                      <p className="text-xs text-muted-foreground">Send anonymous crash reports</p>
                    </div>
                    <Switch id="crash-reports" defaultChecked />
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline">Reset to Defaults</Button>
                <Button>Save Changes</Button>
              </CardFooter>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="cameras" className="mt-6 space-y-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Camera Settings</CardTitle>
                <CardDescription>Configure camera resolution, frame rate, and recording settings</CardDescription>
              </div>
              <Dialog>
                <DialogTrigger asChild>
                  <Button>
                    <Camera className="mr-2 h-4 w-4" /> Add Camera
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[600px]">
                  <DialogHeader>
                    <DialogTitle>Add New Camera</DialogTitle>
                    <DialogDescription>
                      Configure a new camera for your surveillance system.
                    </DialogDescription>
                  </DialogHeader>
                  <AddCameraForm />
                </DialogContent>
              </Dialog>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[200px]">Camera</TableHead>
                    <TableHead>Location</TableHead>
                    <TableHead>Resolution</TableHead>
                    <TableHead>Frame Rate</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {cameraList.map((camera) => (
                    <TableRow key={camera.id}>
                      <TableCell className="font-medium">{camera.name}</TableCell>
                      <TableCell>{camera.location}</TableCell>
                      <TableCell>
                        <Select defaultValue={camera.resolution}>
                          <SelectTrigger className="w-[100px]">
                            <SelectValue placeholder="Resolution" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="720p">720p</SelectItem>
                            <SelectItem value="1080p">1080p</SelectItem>
                            <SelectItem value="4K">4K</SelectItem>
                          </SelectContent>
                        </Select>
                      </TableCell>
                      <TableCell>
                        <Select defaultValue={"30"}>
                          <SelectTrigger className="w-[80px]">
                            <SelectValue placeholder="FPS" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="15">15 FPS</SelectItem>
                            <SelectItem value="30">30 FPS</SelectItem>
                            <SelectItem value="60">60 FPS</SelectItem>
                          </SelectContent>
                        </Select>
                      </TableCell>
                      <TableCell>
                        {camera.status === true ? (
                          <Badge className="bg-green-500">Online</Badge>
                        ) : (
                          <Badge variant="outline" className="border-red-500 text-red-500">
                            Offline
                          </Badge>
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-2">
                          <Button variant="outline" size="sm">
                            Edit
                          </Button>
                          <Button variant="outline" size="sm" className="text-red-500">
                            Delete
                          </Button>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>

              </Table>
            </CardContent>
            <CardFooter className="flex justify-between">
              <div className="text-sm text-muted-foreground">
                {cameraList.filter((c) => c.status === true).length} of {cameraList.length} cameras online
              </div>
              <Button>
                <Save className="mr-2 h-4 w-4" /> Save All Changes
              </Button>
            </CardFooter>
          </Card>

          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Recording Settings</CardTitle>
                <CardDescription>Configure when and how cameras record footage</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="continuous-recording">Continuous Recording</Label>
                      <p className="text-xs text-muted-foreground">Record footage continuously</p>
                    </div>
                    <Switch id="continuous-recording" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="motion-detection">Motion Detection</Label>
                      <p className="text-xs text-muted-foreground">Record only when motion is detected</p>
                    </div>
                    <Switch id="motion-detection" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="scheduled-recording">Scheduled Recording</Label>
                      <p className="text-xs text-muted-foreground">Record during specified time periods</p>
                    </div>
                    <Switch id="scheduled-recording" />
                  </div>
                </div>
                <Separator />
                <div className="space-y-2">
                  <Label htmlFor="motion-sensitivity">Motion Sensitivity</Label>
                  <Slider id="motion-sensitivity" defaultValue={[75]} max={100} step={1} />
                  <div className="flex justify-between">
                    <span className="text-xs text-muted-foreground">Low</span>
                    <span className="text-xs text-muted-foreground">High</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="pre-record">Pre-Record Time</Label>
                  <Select defaultValue="10">
                    <SelectTrigger id="pre-record">
                      <SelectValue placeholder="Select pre-record time" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="5">5 seconds</SelectItem>
                      <SelectItem value="10">10 seconds</SelectItem>
                      <SelectItem value="30">30 seconds</SelectItem>
                      <SelectItem value="60">1 minute</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Amount of footage to save before motion is detected</p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="post-record">Post-Record Time</Label>
                  <Select defaultValue="30">
                    <SelectTrigger id="post-record">
                      <SelectValue placeholder="Select post-record time" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="10">10 seconds</SelectItem>
                      <SelectItem value="30">30 seconds</SelectItem>
                      <SelectItem value="60">1 minute</SelectItem>
                      <SelectItem value="300">5 minutes</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">Amount of footage to save after motion stops</p>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline">Reset to Defaults</Button>
                <Button>Save Changes</Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Video Quality Settings</CardTitle>
                <CardDescription>Configure video quality and compression settings</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="video-quality">Default Video Quality</Label>
                  <Select defaultValue="high">
                    <SelectTrigger id="video-quality">
                      <SelectValue placeholder="Select video quality" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low (720p, 15fps)</SelectItem>
                      <SelectItem value="medium">Medium (1080p, 15fps)</SelectItem>
                      <SelectItem value="high">High (1080p, 30fps)</SelectItem>
                      <SelectItem value="ultra">Ultra (4K, 30fps)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="compression">Compression Level</Label>
                  <Slider id="compression" defaultValue={[30]} max={100} step={1} />
                  <div className="flex justify-between">
                    <span className="text-xs text-muted-foreground">Higher Quality</span>
                    <span className="text-xs text-muted-foreground">Smaller Size</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="bitrate">Bitrate (Mbps)</Label>
                  <Select defaultValue="8">
                    <SelectTrigger id="bitrate">
                      <SelectValue placeholder="Select bitrate" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="2">2 Mbps</SelectItem>
                      <SelectItem value="4">4 Mbps</SelectItem>
                      <SelectItem value="8">8 Mbps</SelectItem>
                      <SelectItem value="16">16 Mbps</SelectItem>
                      <SelectItem value="32">32 Mbps</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Separator />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="audio-recording">Audio Recording</Label>
                      <p className="text-xs text-muted-foreground">Record audio with video footage</p>
                    </div>
                    <Switch id="audio-recording" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="timestamp">Show Timestamp</Label>
                      <p className="text-xs text-muted-foreground">Display date and time on footage</p>
                    </div>
                    <Switch id="timestamp" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="camera-name">Show Camera Name</Label>
                      <p className="text-xs text-muted-foreground">Display camera name on footage</p>
                    </div>
                    <Switch id="camera-name" defaultChecked />
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline">Reset to Defaults</Button>
                <Button>Save Changes</Button>
              </CardFooter>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="storage" className="mt-6 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Storage Management</CardTitle>
                <CardDescription>Configure storage capacity and retention policies</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Storage Usage</Label>
                    <Badge variant="outline">
                      {usedStorage.toFixed(4)} GB / {totalStorage} GB
                    </Badge>
                  </div>
                  <Progress value={(usedStorage / totalStorage) * 100} className="h-2" />
                  <p className="text-xs text-muted-foreground">
                    {(((totalStorage - usedStorage) / totalStorage) * 100).toFixed(4)}% free space available
                  </p>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="retention-days">Retention Period ({retentionDays[0]} days)</Label>
                  <Slider
                    id="retention-days"
                    min={7}
                    max={90}
                    step={1}
                    value={retentionDays}
                    onValueChange={handleRetentionChange}
                  />
                  <p className="text-xs text-muted-foreground">
                    Footage older than {retentionDays[0]} days will be automatically deleted
                  </p>
                </div>
                <Separator />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="auto-delete">Auto Delete Old Footage</Label>
                      <p className="text-xs text-muted-foreground">
                        Automatically delete footage older than retention period
                      </p>
                    </div>
                    <Switch id="auto-delete" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="low-space-alert">Low Space Alert</Label>
                      <p className="text-xs text-muted-foreground">Send alert when storage space is low</p>
                    </div>
                    <Switch id="low-space-alert" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="low-space-threshold">Low Space Threshold</Label>
                    <Select defaultValue="10">
                      <SelectTrigger id="low-space-threshold" className="w-[100px]">
                        <SelectValue placeholder="Select threshold" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="5">5%</SelectItem>
                        <SelectItem value="10">10%</SelectItem>
                        <SelectItem value="15">15%</SelectItem>
                        <SelectItem value="20">20%</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button variant="outline" className="gap-2">
                        <Trash2 className="h-4 w-4" /> Clear All Footage
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>This action cannot be undone</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                <Button>Save Changes</Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Storage Usage</CardTitle>
                <CardDescription>View storage usage by camera and date</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <StorageChart />
                <div className="space-y-2">
                  <Label>Storage by Camera</Label>
                  <div className="space-y-2">
                    {cameraList
                      .filter((c) => c.status === true)
                      .sort((a, b) => b.storage - a.storage)
                      .map((camera) => (
                        <div key={camera.id} className="space-y-1">
                          <div className="flex items-center justify-between">
                            <span className="text-sm">{camera.name}</span>
                            <span className="text-sm font-medium">{camera.storage.toFixed(4)} GB</span>
                          </div>
                          <Progress value={(camera.storage / usedStorage) * 100} className="h-1" />
                        </div>
                      ))}
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <div className="flex w-full items-center justify-between rounded-md bg-muted p-3">
                  <div className="flex items-center gap-2">
                    <Info className="h-4 w-4 text-blue-500" />
                    <span className="text-sm">Storage usage is updated hourly</span>
                  </div>
                </div>
              </CardFooter>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Storage Locations</CardTitle>
              <CardDescription>Configure where footage is stored</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="local-storage">Local Storage</Label>
                    <p className="text-xs text-muted-foreground">Store footage on local drives</p>
                  </div>
                  <Switch id="local-storage" defaultChecked />
                </div>
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label htmlFor="cloud-backup">Cloud Backup</Label>
                    <p className="text-xs text-muted-foreground">Backup footage to cloud storage</p>
                  </div>
                  <Switch id="cloud-backup" />
                </div>
                <div className="flex items-center justify-between">
                  <Label htmlFor="cloud-provider">Cloud Provider</Label>
                  <Select defaultValue="aws">
                    <SelectTrigger id="cloud-provider" className="w-[180px]">
                      <SelectValue placeholder="Select provider" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="aws">Amazon S3</SelectItem>
                      <SelectItem value="azure">Azure Blob Storage</SelectItem>
                      <SelectItem value="gcp">Google Cloud Storage</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              <div className="rounded-md bg-muted p-3">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="mt-0.5 h-5 w-5 text-yellow-500" />
                  <div className="space-y-1">
                    <p className="font-medium">Cloud Storage Not Configured</p>
                    <p className="text-sm text-muted-foreground">
                      Cloud backup is enabled but not configured. Please configure your cloud storage settings.
                    </p>
                    <Button size="sm" variant="outline" className="mt-2">
                      Configure Cloud Storage
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline">Reset to Defaults</Button>
              <Button>Save Changes</Button>
            </CardFooter>
          </Card>
        </TabsContent>
        <TabsContent value="notifications" className="mt-6 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Notification Settings</CardTitle>
                <CardDescription>Configure how and when you receive notifications</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="email-notifications">Email Notifications</Label>
                      <p className="text-xs text-muted-foreground">Receive notifications via email</p>
                    </div>
                    <Switch id="email-notifications" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="push-notifications">Push Notifications</Label>
                      <p className="text-xs text-muted-foreground">Receive notifications on your device</p>
                    </div>
                    <Switch id="push-notifications" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="sms-notifications">SMS Notifications</Label>
                      <p className="text-xs text-muted-foreground">Receive notifications via SMS</p>
                    </div>
                    <Switch id="sms-notifications" />
                  </div>
                </div>
                <Separator />
                <div className="space-y-2">
                  <Label>Notification Frequency</Label>
                  <Select defaultValue="immediate">
                    <SelectTrigger>
                      <SelectValue placeholder="Select frequency" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="immediate">Immediate</SelectItem>
                      <SelectItem value="hourly">Hourly Digest</SelectItem>
                      <SelectItem value="daily">Daily Digest</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Quiet Hours</Label>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="quiet-start">Start Time</Label>
                      <Select defaultValue="22">
                        <SelectTrigger id="quiet-start">
                          <SelectValue placeholder="Select time" />
                        </SelectTrigger>
                        <SelectContent>
                          {Array.from({ length: 24 }).map((_, i) => (
                            <SelectItem key={i} value={i.toString()}>
                              {i === 0 ? "12 AM" : i < 12 ? `${i} AM` : i === 12 ? "12 PM" : `${i - 12} PM`}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="quiet-end">End Time</Label>
                      <Select defaultValue="7">
                        <SelectTrigger id="quiet-end">
                          <SelectValue placeholder="Select time" />
                        </SelectTrigger>
                        <SelectContent>
                          {Array.from({ length: 24 }).map((_, i) => (
                            <SelectItem key={i} value={i.toString()}>
                              {i === 0 ? "12 AM" : i < 12 ? `${i} AM` : i === 12 ? "12 PM" : `${i - 12} PM`}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground">Only critical alerts will be sent during quiet hours</p>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline">Reset to Defaults</Button>
                <Button>Save Changes</Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Alert Types</CardTitle>
                <CardDescription>Configure which events trigger notifications</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="motion-alerts">Motion Detection</Label>
                      <p className="text-xs text-muted-foreground">Alert when motion is detected</p>
                    </div>
                    <Switch id="motion-alerts" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="person-alerts">Person Detection</Label>
                      <p className="text-xs text-muted-foreground">Alert when a person is detected</p>
                    </div>
                    <Switch id="person-alerts" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="vehicle-alerts">Vehicle Detection</Label>
                      <p className="text-xs text-muted-foreground">Alert when a vehicle is detected</p>
                    </div>
                    <Switch id="vehicle-alerts" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="camera-offline">Camera Offline</Label>
                      <p className="text-xs text-muted-foreground">Alert when a camera goes offline</p>
                    </div>
                    <Switch id="camera-offline" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="storage-alerts">Storage Alerts</Label>
                      <p className="text-xs text-muted-foreground">Alert when storage is running low</p>
                    </div>
                    <Switch id="storage-alerts" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="system-alerts">System Alerts</Label>
                      <p className="text-xs text-muted-foreground">Alert for system issues and updates</p>
                    </div>
                    <Switch id="system-alerts" defaultChecked />
                  </div>
                </div>
              </CardContent>
              <CardFooter className="flex justify-between">
                <Button variant="outline">Reset to Defaults</Button>
                <Button>Save Changes</Button>
              </CardFooter>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Notification Recipients</CardTitle>
                <CardDescription>Configure who receives notifications</CardDescription>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Email</TableHead>
                      <TableHead>Phone</TableHead>
                      <TableHead>Alert Level</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    <TableRow>
                      <TableCell className="font-medium">Admin User</TableCell>
                      <TableCell>admin@example.com</TableCell>
                      <TableCell>+1 (555) 123-4567</TableCell>
                      <TableCell>
                        <Badge>All Alerts</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="sm">
                          Edit
                        </Button>
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">Security Team</TableCell>
                      <TableCell>security@example.com</TableCell>
                      <TableCell>+1 (555) 987-6543</TableCell>
                      <TableCell>
                        <Badge variant="outline">Critical Only</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="sm">
                          Edit
                        </Button>
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell className="font-medium">IT Support</TableCell>
                      <TableCell>it@example.com</TableCell>
                      <TableCell>—</TableCell>
                      <TableCell>
                        <Badge variant="outline">System Alerts</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button variant="ghost" size="sm">
                          Edit
                        </Button>
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </CardContent>
              <CardFooter>
                <Button className="ml-auto">Add Recipient</Button>
              </CardFooter>
            </Card>
          </div>
          </TabsContent>
        </Tabs>
      </div>
    )
  }