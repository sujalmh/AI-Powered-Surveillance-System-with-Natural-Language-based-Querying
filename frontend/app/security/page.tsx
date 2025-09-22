"use client"

import { useState } from "react"
import {
  Key,
  Eye,
  EyeOff,
  AlertTriangle,
  RefreshCw,
  Smartphone,
  Mail,
  Globe,
  Clock,
  User,
  FileText,
  CheckCircle,
  XCircle,
  Info,
  Plus,
  Filter,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Progress } from "@/components/ui/progress"
import { SecurityChart } from "@/components/security-chart"

type AuditLogEntry = {
  id: string
  action: string
  user: string
  timestamp: Date
  ip: string
  status: "success" | "failure"
  details?: string
}

const mockAuditLogs: AuditLogEntry[] = [
  {
    id: "log1",
    action: "Login",
    user: "admin@example.com",
    timestamp: new Date(2023, 3, 15, 9, 30),
    ip: "192.168.1.1",
    status: "success",
  },
  {
    id: "log2",
    action: "Password Change",
    user: "john@example.com",
    timestamp: new Date(2023, 3, 15, 10, 15),
    ip: "192.168.1.2",
    status: "success",
  },
  {
    id: "log3",
    action: "Login",
    user: "jane@example.com",
    timestamp: new Date(2023, 3, 15, 11, 20),
    ip: "192.168.1.3",
    status: "success",
  },
  {
    id: "log4",
    action: "Login Attempt",
    user: "robert@example.com",
    timestamp: new Date(2023, 3, 15, 12, 45),
    ip: "192.168.1.4",
    status: "failure",
    details: "Invalid password",
  },
  {
    id: "log5",
    action: "User Created",
    user: "admin@example.com",
    timestamp: new Date(2023, 3, 15, 14, 10),
    ip: "192.168.1.1",
    status: "success",
    details: "Created user emily@example.com",
  },
  {
    id: "log6",
    action: "Settings Changed",
    user: "admin@example.com",
    timestamp: new Date(2023, 3, 15, 15, 30),
    ip: "192.168.1.1",
    status: "success",
    details: "Updated password policy",
  },
  {
    id: "log7",
    action: "Login Attempt",
    user: "unknown",
    timestamp: new Date(2023, 3, 15, 16, 45),
    ip: "203.0.113.1",
    status: "failure",
    details: "User not found",
  },
  {
    id: "log8",
    action: "2FA Enabled",
    user: "david@example.com",
    timestamp: new Date(2023, 3, 15, 17, 20),
    ip: "192.168.1.5",
    status: "success",
  },
]

export default function SecurityPage() {
  const [passwordLength, setPasswordLength] = useState<number[]>([12])
  const [showPassword, setShowPassword] = useState(false)
  const [passwordExample, setPasswordExample] = useState("P@ssw0rd!2023")
  const [passwordStrength, setPasswordStrength] = useState(85)

  const handlePasswordLengthChange = (value: number[]) => {
    setPasswordLength(value)
  }

  const togglePasswordVisibility = () => {
    setShowPassword(!showPassword)
  }

  return (
    <div className="flex flex-col gap-4 sm:gap-6 p-4 sm:p-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Security</h1>
        <p className="text-muted-foreground">Configure system security settings and review audit logs</p>
      </div>

      <Tabs defaultValue="password-policy">
        <TabsList className="grid w-full grid-cols-2 md:grid-cols-4">
          <TabsTrigger value="password-policy">Password Policy</TabsTrigger>
          <TabsTrigger value="two-factor">Two-Factor Auth</TabsTrigger>
          <TabsTrigger value="ip-restrictions">IP Restrictions</TabsTrigger>
          <TabsTrigger value="audit-logs">Audit Logs</TabsTrigger>
        </TabsList>
        <TabsContent value="password-policy" className="mt-6 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Password Requirements</CardTitle>
                <CardDescription>Configure the password requirements for all users</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="password-length">Minimum Length ({passwordLength[0]} characters)</Label>
                    <span className="text-sm text-muted-foreground">Recommended: 12+</span>
                  </div>
                  <Slider
                    id="password-length"
                    min={8}
                    max={24}
                    step={1}
                    value={passwordLength}
                    onValueChange={handlePasswordLengthChange}
                  />
                </div>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="uppercase">Require uppercase letters</Label>
                      <Badge variant="outline" className="text-xs">
                        A-Z
                      </Badge>
                    </div>
                    <Switch id="uppercase" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="lowercase">Require lowercase letters</Label>
                      <Badge variant="outline" className="text-xs">
                        a-z
                      </Badge>
                    </div>
                    <Switch id="lowercase" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="numbers">Require numbers</Label>
                      <Badge variant="outline" className="text-xs">
                        0-9
                      </Badge>
                    </div>
                    <Switch id="numbers" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Label htmlFor="special">Require special characters</Label>
                      <Badge variant="outline" className="text-xs">
                        !@#$%
                      </Badge>
                    </div>
                    <Switch id="special" defaultChecked />
                  </div>
                </div>
                <Separator />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="expiration">Password expiration</Label>
                    <Select defaultValue="90">
                      <SelectTrigger id="expiration" className="w-[180px]">
                        <SelectValue placeholder="Select expiration" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="30">30 days</SelectItem>
                        <SelectItem value="60">60 days</SelectItem>
                        <SelectItem value="90">90 days</SelectItem>
                        <SelectItem value="180">180 days</SelectItem>
                        <SelectItem value="never">Never</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center justify-between">
                    <Label htmlFor="history">Password history</Label>
                    <Select defaultValue="5">
                      <SelectTrigger id="history" className="w-[180px]">
                        <SelectValue placeholder="Select history" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="3">Remember 3 passwords</SelectItem>
                        <SelectItem value="5">Remember 5 passwords</SelectItem>
                        <SelectItem value="10">Remember 10 passwords</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-1">
                      <Label htmlFor="lockout">Account lockout</Label>
                      <p className="text-xs text-muted-foreground">After failed login attempts</p>
                    </div>
                    <Select defaultValue="5">
                      <SelectTrigger id="lockout" className="w-[180px]">
                        <SelectValue placeholder="Select threshold" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="3">3 attempts</SelectItem>
                        <SelectItem value="5">5 attempts</SelectItem>
                        <SelectItem value="10">10 attempts</SelectItem>
                      </SelectContent>
                    </Select>
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
                <CardTitle>Password Strength Tester</CardTitle>
                <CardDescription>Test password strength against current policy</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="password-test">Test Password</Label>
                  <div className="relative">
                    <Input
                      id="password-test"
                      type={showPassword ? "text" : "password"}
                      value={passwordExample}
                      onChange={(e) => setPasswordExample(e.target.value)}
                      className="pr-10"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="absolute right-0 top-0 h-full"
                      onClick={togglePasswordVisibility}
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Password Strength</Label>
                    <span className="text-sm font-medium">
                      {passwordStrength < 40 ? "Weak" : passwordStrength < 70 ? "Moderate" : "Strong"}
                    </span>
                  </div>
                  <Progress value={passwordStrength} className="h-2" />
                </div>
                <div className="space-y-2">
                  <Label>Requirements Check</Label>
                  <div className="space-y-2 rounded-md border p-3">
                    <div className="flex items-center gap-2">
                      {passwordExample.length >= passwordLength[0] ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      <span className="text-sm">Minimum length ({passwordLength[0]} characters)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {/[A-Z]/.test(passwordExample) ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      <span className="text-sm">Contains uppercase letters</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {/[a-z]/.test(passwordExample) ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      <span className="text-sm">Contains lowercase letters</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {/[0-9]/.test(passwordExample) ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      <span className="text-sm">Contains numbers</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {/[^A-Za-z0-9]/.test(passwordExample) ? (
                        <CheckCircle className="h-4 w-4 text-green-500" />
                      ) : (
                        <XCircle className="h-4 w-4 text-red-500" />
                      )}
                      <span className="text-sm">Contains special characters</span>
                    </div>
                  </div>
                </div>
              </CardContent>
              <CardFooter>
                <div className="flex w-full items-center justify-between rounded-md bg-muted p-3">
                  <div className="flex items-center gap-2">
                    <Info className="h-4 w-4 text-blue-500" />
                    <span className="text-sm">Passwords should be unique and not used on other sites</span>
                  </div>
                </div>
              </CardFooter>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Security Metrics</CardTitle>
              <CardDescription>Overview of security events and password-related metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <SecurityChart />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="two-factor" className="mt-6 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Two-Factor Authentication</CardTitle>
                <CardDescription>Configure two-factor authentication settings for users</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="require-2fa">Require 2FA for all users</Label>
                      <p className="text-xs text-muted-foreground">
                        All users will be required to set up 2FA on next login
                      </p>
                    </div>
                    <Switch id="require-2fa" />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="require-admin-2fa">Require 2FA for admins</Label>
                      <p className="text-xs text-muted-foreground">All admin users will be required to set up 2FA</p>
                    </div>
                    <Switch id="require-admin-2fa" defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="remember-device">Remember device</Label>
                      <p className="text-xs text-muted-foreground">Allow users to remember devices for 2FA</p>
                    </div>
                    <Switch id="remember-device" defaultChecked />
                  </div>
                </div>
                <Separator />
                <div className="space-y-4">
                  <Label>Allowed 2FA Methods</Label>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Smartphone className="h-5 w-5 text-primary" />
                        <div>
                          <p className="font-medium">Authenticator App</p>
                          <p className="text-sm text-muted-foreground">
                            Google Authenticator, Microsoft Authenticator, etc.
                          </p>
                        </div>
                      </div>
                      <Switch defaultChecked />
                    </div>
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Mail className="h-5 w-5 text-primary" />
                        <div>
                          <p className="font-medium">Email</p>
                          <p className="text-sm text-muted-foreground">Send verification code via email</p>
                        </div>
                      </div>
                      <Switch defaultChecked />
                    </div>
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Key className="h-5 w-5 text-primary" />
                        <div>
                          <p className="font-medium">Security Keys</p>
                          <p className="text-sm text-muted-foreground">YubiKey, Google Titan, etc.</p>
                        </div>
                      </div>
                      <Switch />
                    </div>
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
                <CardTitle>2FA Status</CardTitle>
                <CardDescription>Overview of two-factor authentication usage</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2 rounded-md border p-4">
                    <p className="text-sm text-muted-foreground">Users with 2FA</p>
                    <div className="flex items-end justify-between">
                      <p className="text-2xl font-bold">5</p>
                      <Badge className="bg-green-500">62.5%</Badge>
                    </div>
                  </div>
                  <div className="space-y-2 rounded-md border p-4">
                    <p className="text-sm text-muted-foreground">Users without 2FA</p>
                    <div className="flex items-end justify-between">
                      <p className="text-2xl font-bold">3</p>
                      <Badge variant="outline" className="border-red-500 text-red-500">
                        37.5%
                      </Badge>
                    </div>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>2FA Methods Used</Label>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Smartphone className="h-5 w-5 text-primary" />
                        <span>Authenticator App</span>
                      </div>
                      <Badge>4 users</Badge>
                    </div>
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Mail className="h-5 w-5 text-primary" />
                        <span>Email</span>
                      </div>
                      <Badge>1 user</Badge>
                    </div>
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Key className="h-5 w-5 text-primary" />
                        <span>Security Keys</span>
                      </div>
                      <Badge>0 users</Badge>
                    </div>
                  </div>
                </div>
                <div className="rounded-md bg-muted p-3">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="mt-0.5 h-5 w-5 text-yellow-500" />
                    <div className="space-y-1">
                      <p className="font-medium">Security Recommendation</p>
                      <p className="text-sm text-muted-foreground">
                        Enforce 2FA for all users to improve security. 3 users still don't have 2FA enabled.
                      </p>
                      <Button size="sm" variant="outline" className="mt-2">
                        Enforce Now
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
        <TabsContent value="ip-restrictions" className="mt-6 space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>IP Address Restrictions</CardTitle>
                <CardDescription>Configure IP address restrictions for system access</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="enable-ip-restrictions">Enable IP restrictions</Label>
                      <p className="text-xs text-muted-foreground">
                        Restrict access to specific IP addresses or ranges
                      </p>
                    </div>
                    <Switch id="enable-ip-restrictions" />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="admin-ip-restrictions">Admin IP restrictions</Label>
                      <p className="text-xs text-muted-foreground">Apply stricter IP restrictions for admin users</p>
                    </div>
                    <Switch id="admin-ip-restrictions" defaultChecked />
                  </div>
                </div>
                <Separator />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>IP Whitelist</Label>
                    <Button variant="outline" size="sm">
                      <Plus className="mr-2 h-4 w-4" /> Add IP
                    </Button>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Globe className="h-5 w-5 text-primary" />
                        <div>
                          <p className="font-medium">192.168.1.0/24</p>
                          <p className="text-sm text-muted-foreground">Office Network</p>
                        </div>
                      </div>
                      <Button variant="ghost" size="icon">
                        <XCircle className="h-4 w-4" />
                      </Button>
                    </div>
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Globe className="h-5 w-5 text-primary" />
                        <div>
                          <p className="font-medium">10.0.0.0/8</p>
                          <p className="text-sm text-muted-foreground">VPN Network</p>
                        </div>
                      </div>
                      <Button variant="ghost" size="icon">
                        <XCircle className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>IP Blacklist</Label>
                    <Button variant="outline" size="sm">
                      <Plus className="mr-2 h-4 w-4" /> Add IP
                    </Button>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Globe className="h-5 w-5 text-red-500" />
                        <div>
                          <p className="font-medium">203.0.113.0/24</p>
                          <p className="text-sm text-muted-foreground">Known malicious range</p>
                        </div>
                      </div>
                      <Button variant="ghost" size="icon">
                        <XCircle className="h-4 w-4" />
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

            <Card>
              <CardHeader>
                <CardTitle>Geolocation Restrictions</CardTitle>
                <CardDescription>Restrict access based on geographic location</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label htmlFor="enable-geo-restrictions">Enable geolocation restrictions</Label>
                      <p className="text-xs text-muted-foreground">Restrict access to specific countries or regions</p>
                    </div>
                    <Switch id="enable-geo-restrictions" />
                  </div>
                </div>
                <Separator />
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>Allowed Countries</Label>
                    <Select defaultValue="allowlist">
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Select mode" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="allowlist">Allow selected</SelectItem>
                        <SelectItem value="blocklist">Block selected</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Globe className="h-5 w-5 text-primary" />
                        <span>United States</span>
                      </div>
                      <Button variant="ghost" size="icon">
                        <XCircle className="h-4 w-4" />
                      </Button>
                    </div>
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Globe className="h-5 w-5 text-primary" />
                        <span>Canada</span>
                      </div>
                      <Button variant="ghost" size="icon">
                        <XCircle className="h-4 w-4" />
                      </Button>
                    </div>
                    <div className="flex items-center justify-between rounded-md border p-3">
                      <div className="flex items-center gap-3">
                        <Globe className="h-5 w-5 text-primary" />
                        <span>United Kingdom</span>
                      </div>
                      <Button variant="ghost" size="icon">
                        <XCircle className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                  <Button variant="outline" size="sm">
                    <Plus className="mr-2 h-4 w-4" /> Add Country
                  </Button>
                </div>
                <div className="rounded-md bg-muted p-3">
                  <div className="flex items-start gap-3">
                    <Info className="mt-0.5 h-5 w-5 text-blue-500" />
                    <div className="space-y-1">
                      <p className="font-medium">Geolocation Accuracy</p>
                      <p className="text-sm text-muted-foreground">
                        Geolocation is based on IP address and may not be 100% accurate. Consider using additional
                        authentication methods for sensitive operations.
                      </p>
                    </div>
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
        <TabsContent value="audit-logs" className="mt-6 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Audit Logs</CardTitle>
              <CardDescription>Review system activity and security events</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div className="flex items-center gap-2">
                    <Input placeholder="Search logs..." className="max-w-sm" />
                    <Button variant="outline" size="icon">
                      <Filter className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Select defaultValue="all">
                      <SelectTrigger className="w-[150px]">
                        <SelectValue placeholder="Filter by action" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Actions</SelectItem>
                        <SelectItem value="login">Login</SelectItem>
                        <SelectItem value="settings">Settings</SelectItem>
                        <SelectItem value="user">User Management</SelectItem>
                      </SelectContent>
                    </Select>
                    <Select defaultValue="all">
                      <SelectTrigger className="w-[150px]">
                        <SelectValue placeholder="Filter by status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Statuses</SelectItem>
                        <SelectItem value="success">Success</SelectItem>
                        <SelectItem value="failure">Failure</SelectItem>
                      </SelectContent>
                    </Select>
                    <Button variant="outline">
                      <RefreshCw className="mr-2 h-4 w-4" /> Refresh
                    </Button>
                  </div>
                </div>
                <div className="overflow-auto rounded-md border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-[100px]">Time</TableHead>
                        <TableHead className="w-[150px]">Action</TableHead>
                        <TableHead className="w-[150px]">User</TableHead>
                        <TableHead className="w-[120px]">IP Address</TableHead>
                        <TableHead>Details</TableHead>
                        <TableHead className="w-[100px]">Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {mockAuditLogs.map((log) => (
                        <TableRow key={log.id}>
                          <TableCell className="font-mono text-xs">
                            <div className="flex items-center gap-1">
                              <Clock className="h-3 w-3 text-muted-foreground" />
                              <span>
                                {log.timestamp.toLocaleDateString()}{" "}
                                {log.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                              </span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant="outline"
                              className={
                                log.action.includes("Login")
                                  ? "border-blue-500 text-blue-500"
                                  : log.action.includes("Password")
                                    ? "border-purple-500 text-purple-500"
                                    : log.action.includes("User")
                                      ? "border-green-500 text-green-500"
                                      : "border-orange-500 text-orange-500"
                              }
                            >
                              {log.action}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex items-center gap-1">
                              <User className="h-3 w-3 text-muted-foreground" />
                              <span>{log.user}</span>
                            </div>
                          </TableCell>
                          <TableCell className="font-mono text-xs">{log.ip}</TableCell>
                          <TableCell>{log.details || <span className="text-muted-foreground">—</span>}</TableCell>
                          <TableCell>
                            {log.status === "success" ? (
                              <Badge className="bg-green-500">Success</Badge>
                            ) : (
                              <Badge variant="destructive">Failure</Badge>
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
                <div className="flex items-center justify-between">
                  <div className="text-sm text-muted-foreground">Showing 8 of 256 entries</div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" disabled>
                      Previous
                    </Button>
                    <Button variant="outline" size="sm">
                      Next
                    </Button>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline">
                <FileText className="mr-2 h-4 w-4" /> Export Logs
              </Button>
              <Button variant="destructive">Clear Logs</Button>
            </CardFooter>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
