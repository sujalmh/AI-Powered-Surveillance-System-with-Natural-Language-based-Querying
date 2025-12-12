"use client"

import { Bell, User, Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { useSidebar } from "@/lib/sidebar-context"
import { useState, useEffect } from "react"

export function Header() {
  const { theme, setTheme } = useTheme()
  const { hovered } = useSidebar()
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <header
      className={`h-14 glass-header glass-noise flex items-center justify-between px-4 z-40 transition-all duration-300 ease-in-out overflow-hidden shrink-0 w-full border-b`}
    >
      {/* Bottom accent line */}
      <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-accent/30 to-transparent" />
      
      {/* Corner accent glow */}
      <div className="absolute top-0 right-0 w-32 h-full bg-gradient-to-l from-accent/5 to-transparent pointer-events-none" />
      
      <div className="flex-1 relative z-10" />

      <div className="flex items-center gap-6">
        <button className="p-2.5 hover:bg-sidebar-accent/10 rounded-lg transition-all duration-300 hover:scale-110 active:scale-95 relative group">
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-sidebar-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg" />
          <Bell className="w-5 h-5 text-sidebar-foreground relative z-10" />
        </button>

        <button
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          className="p-2.5 hover:bg-sidebar-accent/10 rounded-lg transition-all duration-300 hover:scale-110 active:scale-95 relative group"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-sidebar-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg" />
          {mounted ? (
            theme === "dark" ? (
              <Sun className="w-5 h-5 text-sidebar-foreground relative z-10" />
            ) : (
              <Moon className="w-5 h-5 text-sidebar-foreground relative z-10" />
            )
          ) : (
            <Moon className="w-5 h-5 text-sidebar-foreground relative z-10" />
          )}
        </button>

        <div className="flex items-center gap-4 pl-6 ml-2 border-l border-border/50">
          <div className="text-right">
            <p className="text-sm font-semibold text-foreground">Admin User</p>
            <p className="text-xs text-muted-foreground">System Admin</p>
          </div>
          <div className="w-11 h-11 rounded-full gradient-button flex items-center justify-center shadow-lg hover:shadow-xl transition-shadow duration-300">
            <User className="w-5 h-5 text-primary-foreground" />
          </div>
        </div>
      </div>
    </header>
  )
}
