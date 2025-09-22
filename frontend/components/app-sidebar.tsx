"use client"

import { usePathname, useRouter } from "next/navigation"
import { BarChart3, Camera, Home, MessageSquare, Settings, Bell, Shield, Users } from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarSeparator,
  SidebarTrigger,
  SidebarRail,
} from "@/components/ui/sidebar"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"

export function AppSidebar() {
  const router = useRouter()
  const pathname = usePathname()

  const mainNavItems = [
    {
      title: "Dashboard",
      icon: Home,
      href: "/",
    },
    {
      title: "AI Chat",
      icon: MessageSquare,
      href: "/chat",
      featured: true, // Add this property
    },
    {
      title: "Camera Feeds",
      icon: Camera,
      href: "/cameras",
    },
    {
      title: "Analytics",
      icon: BarChart3,
      href: "/analytics",
    },
  ]

  const secondaryNavItems = [
    {
      title: "Alerts",
      icon: Bell,
      href: "/alerts",
    },
    {
      title: "Users",
      icon: Users,
      href: "/users",
    },
    {
      title: "Security",
      icon: Shield,
      href: "/security",
    },
    {
      title: "Settings",
      icon: Settings,
      href: "/settings",
    },
  ]

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader className="flex items-center justify-between p-4">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-md bg-primary">
            <Shield className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="text-lg font-semibold">SecureView AI</span>
        </div>
        <SidebarTrigger />
      </SidebarHeader>
      <SidebarSeparator />
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Main</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {mainNavItems.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    asChild
                    isActive={pathname === item.href}
                    onClick={() => router.push(item.href)}
                    tooltip={item.title}
                    className={item.featured ? "bg-primary/10" : ""}
                  >
                    <button>
                      <item.icon className={`h-5 w-5 ${item.featured ? "text-primary" : ""}`} />
                      <span>{item.title}</span>
                      {item.title === "AI Chat" && (
                        <Badge className="ml-auto bg-primary text-primary-foreground">New</Badge>
                      )}
                    </button>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
        <SidebarSeparator />
        <SidebarGroup>
          <SidebarGroupLabel>System</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {secondaryNavItems.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    asChild
                    isActive={pathname === item.href}
                    onClick={() => router.push(item.href)}
                    tooltip={item.title}
                  >
                    <button>
                      <item.icon className="h-5 w-5" />
                      <span>{item.title}</span>
                    </button>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="p-4">
        <div className="flex items-center gap-3 cursor-pointer" onClick={() => router.push("/profile")}>
          <Avatar>
            <AvatarImage src="/placeholder.svg?height=40&width=40" />
            <AvatarFallback>AD</AvatarFallback>
          </Avatar>
          <div className="flex flex-col">
            <span className="text-sm font-medium">Admin User</span>
            <span className="text-xs text-muted-foreground">Security Admin</span>
          </div>
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
