"use client"

import { createContext, useContext, useState, type ReactNode } from "react"

interface SidebarContextType {
  expanded: boolean
  setExpanded: (expanded: boolean) => void
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined)

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [expanded] = useState(true)

  return (
    <SidebarContext.Provider value={{ expanded, setExpanded: () => {} }}>
      {children}
    </SidebarContext.Provider>
  )
}

export function useSidebar() {
  const context = useContext(SidebarContext)
  if (context === undefined) {
    throw new Error("useSidebar must be used within a SidebarProvider")
  }
  return context
}
