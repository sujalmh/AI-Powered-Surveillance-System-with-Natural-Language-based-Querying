"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { BarChart3, MessageSquare, AlertCircle, Settings, Home, Brain, MonitorCheck } from "lucide-react"
import { useSidebar } from "@/lib/sidebar-context"

export function Sidebar() {
  const pathname = usePathname()
  const { hovered, setHovered } = useSidebar()

  const navItems = [
    { href: "/", label: "Dashboard", icon: Home },
    { href: "/conversation", label: "Conversation", icon: MessageSquare },
    { href: "/alerts", label: "Alerts", icon: AlertCircle },
    { href: "/analytics", label: "Analytics", icon: BarChart3 },
    { href: "/settings", label: "Settings", icon: Settings },
    { href: "/test-dashboard", label: "Test Dashboard", icon: MonitorCheck },
  ]

  const expanded = hovered
  return (
    <aside
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`h-full transition-all duration-300 ease-in-out z-50 glass-sidebar glass-noise border-r overflow-hidden flex-shrink-0 ${
        expanded ? "w-64" : "w-16"
      }`}
    >
      {/* Vertical accent line */}
      <div className="absolute top-0 right-0 bottom-0 w-[2px] bg-gradient-to-b from-transparent via-accent/40 to-transparent" />
      
      {/* Top corner accent */}
      <div className="absolute top-0 left-0 w-20 h-20 bg-gradient-to-br from-accent/10 to-transparent pointer-events-none" />
      
      <div className="p-3 relative z-10">
        {/* Logo section */}
        <div className="flex items-center gap-3 mb-6 h-10 overflow-hidden">
          <div className="w-10 h-10 rounded-xl gradient-button flex items-center justify-center flex-shrink-0 shadow-lg">
            <Brain className="w-6 h-6 text-primary-foreground" />
          </div>
          <h1 
            className={`text-xl font-semibold text-sidebar-foreground whitespace-nowrap transition-opacity duration-300 ${
              expanded ? "opacity-100" : "opacity-0"
            }`}
          >
            SurveillanceAI
          </h1>
        </div>

        {/* Navigation */}
        <nav className="space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.href
            return (
              <Link
                key={item.href}
                href={item.href}
                title={!expanded ? item.label : ""}
                className={`group relative flex items-center gap-3 px-3 py-2 rounded-lg transition-colors overflow-hidden ${
                  isActive
                    ? "border-l-2 border-[var(--accent)] bg-[color-mix(in oklab,var(--accent) 10%,transparent)] text-[color:var(--accent)]"
                    : "text-[color-mix(in oklab,var(--foreground) 60%,transparent)] hover:bg-[color-mix(in oklab,var(--accent) 10%,transparent)] hover:text-[color:var(--foreground)]"
                }`}
              >
                {!isActive && (
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-sidebar-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                )}
                <Icon className={`w-5 h-5 flex-shrink-0 relative z-10 ${isActive ? "text-[var(--accent)]" : "text-slate-500 dark:text-slate-400"}`} />
                <span 
                  className={`font-medium whitespace-nowrap transition-opacity duration-300 relative z-10 ${
                    expanded ? "opacity-100" : "opacity-0"
                  }`}
                >
                  {item.label}
                </span>
              </Link>
            )
          })}
        </nav>
      </div>
    </aside>
  )
}
