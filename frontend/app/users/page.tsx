"use client"

import type React from "react"

import { useState } from "react"
import { Search, Filter, Plus, MoreHorizontal, UserPlus, UserMinus, UserCog, Shield, Clock, Mail } from "lucide-react"
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Label } from "@/components/ui/label"
import { Checkbox } from "@/components/ui/checkbox"
import { UsersChart } from "@/components/users-chart"

type UserRole = "admin" | "operator" | "viewer"

type User = {
  id: string
  name: string
  email: string
  role: UserRole
  status: "active" | "inactive"
  lastLogin: Date | null
  createdAt: Date
  avatar?: string
  department?: string
}

const mockUsers: User[] = [
  {
    id: "u1",
    name: "Admin User",
    email: "admin@example.com",
    role: "admin",
    status: "active",
    lastLogin: new Date(2023, 3, 15, 9, 30),
    createdAt: new Date(2022, 1, 15),
    department: "Security",
  },
  {
    id: "u2",
    name: "John Doe",
    email: "john@example.com",
    role: "operator",
    status: "active",
    lastLogin: new Date(2023, 3, 14, 16, 45),
    createdAt: new Date(2022, 3, 10),
    department: "Operations",
  },
  {
    id: "u3",
    name: "Jane Smith",
    email: "jane@example.com",
    role: "viewer",
    status: "active",
    lastLogin: new Date(2023, 3, 15, 11, 20),
    createdAt: new Date(2022, 5, 22),
    department: "Management",
  },
  {
    id: "u4",
    name: "Robert Johnson",
    email: "robert@example.com",
    role: "operator",
    status: "inactive",
    lastLogin: new Date(2023, 3, 10, 8, 15),
    createdAt: new Date(2022, 7, 5),
    department: "Operations",
  },
  {
    id: "u5",
    name: "Emily Davis",
    email: "emily@example.com",
    role: "viewer",
    status: "active",
    lastLogin: new Date(2023, 3, 15, 14, 10),
    createdAt: new Date(2022, 9, 18),
    department: "HR",
  },
  {
    id: "u6",
    name: "Michael Wilson",
    email: "michael@example.com",
    role: "operator",
    status: "active",
    lastLogin: new Date(2023, 3, 14, 17, 30),
    createdAt: new Date(2022, 11, 3),
    department: "IT",
  },
  {
    id: "u7",
    name: "Sarah Brown",
    email: "sarah@example.com",
    role: "viewer",
    status: "inactive",
    lastLogin: new Date(2023, 3, 5, 10, 45),
    createdAt: new Date(2023, 0, 12),
    department: "Finance",
  },
  {
    id: "u8",
    name: "David Miller",
    email: "david@example.com",
    role: "admin",
    status: "active",
    lastLogin: new Date(2023, 3, 15, 8, 0),
    createdAt: new Date(2022, 2, 28),
    department: "Security",
  },
]

const getRoleBadge = (role: UserRole) => {
  switch (role) {
    case "admin":
      return <Badge className="bg-red-500">Admin</Badge>
    case "operator":
      return <Badge className="bg-blue-500">Operator</Badge>
    case "viewer":
      return <Badge variant="outline">Viewer</Badge>
  }
}

const getStatusBadge = (status: string) => {
  switch (status) {
    case "active":
      return (
        <Badge variant="outline" className="border-green-500 text-green-500">
          Active
        </Badge>
      )
    case "inactive":
      return (
        <Badge variant="outline" className="border-gray-500 text-gray-500">
          Inactive
        </Badge>
      )
    default:
      return null
  }
}

export default function UsersPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [roleFilter, setRoleFilter] = useState<string>("all")
  const [statusFilter, setStatusFilter] = useState<string>("all")
  const [filteredUsers, setFilteredUsers] = useState<User[]>(mockUsers)
  const [isAddUserDialogOpen, setIsAddUserDialogOpen] = useState(false)
  const [newUser, setNewUser] = useState({
    name: "",
    email: "",
    role: "viewer" as UserRole,
    department: "",
  })

  // Apply filters
  const applyFilters = () => {
    let results = [...mockUsers]

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase()
      results = results.filter(
        (user) =>
          user.name.toLowerCase().includes(query) ||
          user.email.toLowerCase().includes(query) ||
          (user.department && user.department.toLowerCase().includes(query)),
      )
    }

    // Role filter
    if (roleFilter !== "all") {
      results = results.filter((user) => user.role === roleFilter)
    }

    // Status filter
    if (statusFilter !== "all") {
      results = results.filter((user) => user.status === statusFilter)
    }

    setFilteredUsers(results)
  }

  // Handle search input change
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value)
    applyFilters()
  }

  // Handle role filter change
  const handleRoleFilterChange = (value: string) => {
    setRoleFilter(value)
    applyFilters()
  }

  // Handle status filter change
  const handleStatusFilterChange = (value: string) => {
    setStatusFilter(value)
    applyFilters()
  }

  // Handle add user form submission
  const handleAddUser = (e: React.FormEvent) => {
    e.preventDefault()
    // In a real app, you would add the user to the database
    console.log("Adding user:", newUser)
    setIsAddUserDialogOpen(false)
    // Reset form
    setNewUser({
      name: "",
      email: "",
      role: "viewer",
      department: "",
    })
  }

  return (
    <div className="flex flex-col gap-4 sm:gap-6 p-4 sm:p-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight">Users</h1>
        <p className="text-muted-foreground">Manage user accounts and permissions</p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Users</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockUsers.length}</div>
            <p className="text-xs text-muted-foreground">
              {mockUsers.filter((u) => u.status === "active").length} active users
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Admins</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockUsers.filter((u) => u.role === "admin").length}</div>
            <p className="text-xs text-muted-foreground">Full system access</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Recent Logins</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {mockUsers.filter((u) => u.lastLogin && u.lastLogin.getTime() > Date.now() - 24 * 60 * 60 * 1000).length}
            </div>
            <p className="text-xs text-muted-foreground">In the last 24 hours</p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>User Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <UsersChart />
        </CardContent>
      </Card>

      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-1 items-center gap-2">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                className="pl-10"
                placeholder="Search users..."
                value={searchQuery}
                onChange={handleSearchChange}
              />
            </div>
            <Button variant="outline" size="icon">
              <Filter className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Select value={roleFilter} onValueChange={handleRoleFilterChange}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Filter by role" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Roles</SelectItem>
                <SelectItem value="admin">Admin</SelectItem>
                <SelectItem value="operator">Operator</SelectItem>
                <SelectItem value="viewer">Viewer</SelectItem>
              </SelectContent>
            </Select>

            <Select value={statusFilter} onValueChange={handleStatusFilterChange}>
              <SelectTrigger className="w-[150px]">
                <SelectValue placeholder="Filter by status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="inactive">Inactive</SelectItem>
              </SelectContent>
            </Select>

            <Dialog open={isAddUserDialogOpen} onOpenChange={setIsAddUserDialogOpen}>
              <DialogTrigger asChild>
                <Button className="gap-2">
                  <Plus className="h-4 w-4" /> Add User
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Add New User</DialogTitle>
                  <DialogDescription>Create a new user account with specific permissions.</DialogDescription>
                </DialogHeader>
                <form onSubmit={handleAddUser}>
                  <div className="grid gap-4 py-4">
                    <div className="grid grid-cols-4 items-center gap-4">
                      <Label htmlFor="name" className="text-right">
                        Name
                      </Label>
                      <Input
                        id="name"
                        value={newUser.name}
                        onChange={(e) => setNewUser({ ...newUser, name: e.target.value })}
                        className="col-span-3"
                        required
                      />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                      <Label htmlFor="email" className="text-right">
                        Email
                      </Label>
                      <Input
                        id="email"
                        type="email"
                        value={newUser.email}
                        onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                        className="col-span-3"
                        required
                      />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                      <Label htmlFor="role" className="text-right">
                        Role
                      </Label>
                      <Select
                        value={newUser.role}
                        onValueChange={(value: UserRole) => setNewUser({ ...newUser, role: value })}
                      >
                        <SelectTrigger id="role" className="col-span-3">
                          <SelectValue placeholder="Select role" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="admin">Admin</SelectItem>
                          <SelectItem value="operator">Operator</SelectItem>
                          <SelectItem value="viewer">Viewer</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                      <Label htmlFor="department" className="text-right">
                        Department
                      </Label>
                      <Input
                        id="department"
                        value={newUser.department}
                        onChange={(e) => setNewUser({ ...newUser, department: e.target.value })}
                        className="col-span-3"
                      />
                    </div>
                    <div className="grid grid-cols-4 items-center gap-4">
                      <div className="text-right">
                        <Label>Options</Label>
                      </div>
                      <div className="col-span-3 space-y-2">
                        <div className="flex items-center space-x-2">
                          <Checkbox id="send-email" />
                          <Label htmlFor="send-email">Send welcome email</Label>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Checkbox id="require-reset" defaultChecked />
                          <Label htmlFor="require-reset">Require password reset</Label>
                        </div>
                      </div>
                    </div>
                  </div>
                  <DialogFooter>
                    <Button type="submit">Add User</Button>
                  </DialogFooter>
                </form>
              </DialogContent>
            </Dialog>
          </div>
        </div>

        <Tabs defaultValue="all">
          <TabsList>
            <TabsTrigger value="all">All Users</TabsTrigger>
            <TabsTrigger value="active">Active</TabsTrigger>
            <TabsTrigger value="inactive">Inactive</TabsTrigger>
          </TabsList>
          <TabsContent value="all" className="mt-4">
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[250px]">User</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Department</TableHead>
                      <TableHead>Last Login</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredUsers.map((user) => (
                      <TableRow key={user.id}>
                        <TableCell>
                          <div className="flex items-center gap-3">
                            <Avatar>
                              <AvatarImage src={user.avatar || "/placeholder.svg"} />
                              <AvatarFallback className="bg-primary text-primary-foreground">
                                {user.name
                                  .split(" ")
                                  .map((n) => n[0])
                                  .join("")}
                              </AvatarFallback>
                            </Avatar>
                            <div>
                              <div className="font-medium">{user.name}</div>
                              <div className="text-sm text-muted-foreground">{user.email}</div>
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>{getRoleBadge(user.role)}</TableCell>
                        <TableCell>{user.department || "—"}</TableCell>
                        <TableCell>
                          {user.lastLogin ? (
                            <div className="flex items-center gap-1">
                              <Clock className="h-3 w-3 text-muted-foreground" />
                              <span>
                                {user.lastLogin.toLocaleDateString()}{" "}
                                {user.lastLogin.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                              </span>
                            </div>
                          ) : (
                            <span className="text-muted-foreground">Never</span>
                          )}
                        </TableCell>
                        <TableCell>{getStatusBadge(user.status)}</TableCell>
                        <TableCell className="text-right">
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuLabel>Actions</DropdownMenuLabel>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem>
                                <UserCog className="mr-2 h-4 w-4" /> Edit User
                              </DropdownMenuItem>
                              <DropdownMenuItem>
                                <Shield className="mr-2 h-4 w-4" /> Change Role
                              </DropdownMenuItem>
                              <DropdownMenuItem>
                                <Mail className="mr-2 h-4 w-4" /> Send Reset Link
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              {user.status === "active" ? (
                                <DropdownMenuItem className="text-red-500">
                                  <UserMinus className="mr-2 h-4 w-4" /> Deactivate
                                </DropdownMenuItem>
                              ) : (
                                <DropdownMenuItem className="text-green-500">
                                  <UserPlus className="mr-2 h-4 w-4" /> Activate
                                </DropdownMenuItem>
                              )}
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="active" className="mt-4">
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[250px]">User</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Department</TableHead>
                      <TableHead>Last Login</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredUsers
                      .filter((user) => user.status === "active")
                      .map((user) => (
                        <TableRow key={user.id}>
                          <TableCell>
                            <div className="flex items-center gap-3">
                              <Avatar>
                                <AvatarImage src={user.avatar || "/placeholder.svg"} />
                                <AvatarFallback className="bg-primary text-primary-foreground">
                                  {user.name
                                    .split(" ")
                                    .map((n) => n[0])
                                    .join("")}
                                </AvatarFallback>
                              </Avatar>
                              <div>
                                <div className="font-medium">{user.name}</div>
                                <div className="text-sm text-muted-foreground">{user.email}</div>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>{getRoleBadge(user.role)}</TableCell>
                          <TableCell>{user.department || "—"}</TableCell>
                          <TableCell>
                            {user.lastLogin ? (
                              <div className="flex items-center gap-1">
                                <Clock className="h-3 w-3 text-muted-foreground" />
                                <span>
                                  {user.lastLogin.toLocaleDateString()}{" "}
                                  {user.lastLogin.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                                </span>
                              </div>
                            ) : (
                              <span className="text-muted-foreground">Never</span>
                            )}
                          </TableCell>
                          <TableCell>{getStatusBadge(user.status)}</TableCell>
                          <TableCell className="text-right">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon">
                                  <MoreHorizontal className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuLabel>Actions</DropdownMenuLabel>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem>
                                  <UserCog className="mr-2 h-4 w-4" /> Edit User
                                </DropdownMenuItem>
                                <DropdownMenuItem>
                                  <Shield className="mr-2 h-4 w-4" /> Change Role
                                </DropdownMenuItem>
                                <DropdownMenuItem>
                                  <Mail className="mr-2 h-4 w-4" /> Send Reset Link
                                </DropdownMenuItem>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem className="text-red-500">
                                  <UserMinus className="mr-2 h-4 w-4" /> Deactivate
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="inactive" className="mt-4">
            <Card>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-[250px]">User</TableHead>
                      <TableHead>Role</TableHead>
                      <TableHead>Department</TableHead>
                      <TableHead>Last Login</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filteredUsers
                      .filter((user) => user.status === "inactive")
                      .map((user) => (
                        <TableRow key={user.id}>
                          <TableCell>
                            <div className="flex items-center gap-3">
                              <Avatar>
                                <AvatarImage src={user.avatar || "/placeholder.svg"} />
                                <AvatarFallback className="bg-muted text-muted-foreground">
                                  {user.name
                                    .split(" ")
                                    .map((n) => n[0])
                                    .join("")}
                                </AvatarFallback>
                              </Avatar>
                              <div>
                                <div className="font-medium">{user.name}</div>
                                <div className="text-sm text-muted-foreground">{user.email}</div>
                              </div>
                            </div>
                          </TableCell>
                          <TableCell>{getRoleBadge(user.role)}</TableCell>
                          <TableCell>{user.department || "—"}</TableCell>
                          <TableCell>
                            {user.lastLogin ? (
                              <div className="flex items-center gap-1">
                                <Clock className="h-3 w-3 text-muted-foreground" />
                                <span>
                                  {user.lastLogin.toLocaleDateString()}{" "}
                                  {user.lastLogin.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                                </span>
                              </div>
                            ) : (
                              <span className="text-muted-foreground">Never</span>
                            )}
                          </TableCell>
                          <TableCell>{getStatusBadge(user.status)}</TableCell>
                          <TableCell className="text-right">
                            <DropdownMenu>
                              <DropdownMenuTrigger asChild>
                                <Button variant="ghost" size="icon">
                                  <MoreHorizontal className="h-4 w-4" />
                                </Button>
                              </DropdownMenuTrigger>
                              <DropdownMenuContent align="end">
                                <DropdownMenuLabel>Actions</DropdownMenuLabel>
                                <DropdownMenuSeparator />
                                <DropdownMenuItem>
                                  <UserCog className="mr-2 h-4 w-4" /> Edit User
                                </DropdownMenuItem>
                                <DropdownMenuItem>
                                  <Shield className="mr-2 h-4 w-4" /> Change Role
                                </DropdownMenuItem>
                                <DropdownMenuItem className="text-green-500">
                                  <UserPlus className="mr-2 h-4 w-4" /> Activate
                                </DropdownMenuItem>
                              </DropdownMenuContent>
                            </DropdownMenu>
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
