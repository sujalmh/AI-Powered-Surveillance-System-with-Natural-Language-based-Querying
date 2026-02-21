"use client"

import type React from "react"
import { Sidebar } from "./sidebar"
import { Header } from "./header"
import { SidebarProvider } from "@/lib/sidebar-context"

export function MainLayout({
  children,
  noPadding = false,
  noScroll = false,
}: {
  children: React.ReactNode
  noPadding?: boolean
  noScroll?: boolean
}) {
  return (
    <SidebarProvider>
      <div
        style={{
          display: "flex",
          height: "100vh",
          width: "100%",
          overflow: "hidden",
          background: "var(--color-bg)",
          padding: "12px",
          gap: "var(--gap-island)",
        }}
      >
        <Sidebar />

        {/* Right column: header + main content */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            minWidth: 0,
            overflow: "hidden",
            gap: "var(--gap-island)",
          }}
        >
          <Header />

          {/* Main content island */}
          <main
            style={{
              flex: 1,
              background: "transparent",
              overflow: noScroll ? "hidden" : "auto",
              display: noScroll ? "flex" : "block",
              flexDirection: "column",
            }}
          >
            {noPadding ? (
              children
            ) : (
              <div
                style={{
                  padding: "20px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "var(--gap-section)",
                  maxWidth: "1440px",
                  height: "100%",
                }}
              >
                {children}
              </div>
            )}
          </main>
        </div>
      </div>
    </SidebarProvider>
  )
}
