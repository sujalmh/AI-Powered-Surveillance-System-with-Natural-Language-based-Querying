import type React from "react"
import { SidebarProvider } from "@/components/ui/sidebar"
import { ThemeProvider } from "@/components/theme-provider"
import { AppSidebar } from "@/components/app-sidebar"
import { Toaster } from "@/components/ui/toaster"
import "./globals.css"

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <title>AI Surveillance Dashboard</title>
        <meta name="description" content="AI-powered surveillance system" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      </head>
      <body className="dark">
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false} disableTransitionOnChange>
          <SidebarProvider>
            <div className="flex h-screen w-full overflow-hidden">
              <AppSidebar />
              <main className="flex-1 w-full overflow-auto bg-background">{children}</main>
            </div>
            <Toaster />
          </SidebarProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}

export const metadata = {
      generator: 'v0.dev'
    };
