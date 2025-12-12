"use client"

import { createContext, useContext, useState, type ReactNode } from "react"

interface SidebarContextType {
  hovered: boolean
  setHovered: (hovered: boolean) => void
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined)

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [hovered, setHovered] = useState(false)

  return <SidebarContext.Provider value={{ hovered, setHovered }}>{children}</SidebarContext.Provider>
}

export function useSidebar() {
  const context = useContext(SidebarContext)
  if (context === undefined) {
    throw new Error("useSidebar must be used within a SidebarProvider")
  }
  return context
}
