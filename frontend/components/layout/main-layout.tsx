"use client"

import type React from "react"
import { Sidebar } from "./sidebar"
import { Header } from "./header"
import { SidebarProvider, useSidebar } from "@/lib/sidebar-context"

export function MainLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <div className="flex h-screen w-full overflow-hidden bg-gradient-to-br from-background via-muted to-background">
        <Sidebar />
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
          <Header />
          <main className="flex-1 overflow-y-auto overflow-x-hidden p-4 scroll-smooth">
            {children}
          </main>
        </div>
      </div>
    </SidebarProvider>
  )
}
