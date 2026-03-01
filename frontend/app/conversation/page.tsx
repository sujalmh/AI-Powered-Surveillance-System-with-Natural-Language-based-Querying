"use client"

import { useState } from "react"
import { MainLayout } from "@/components/layout/main-layout"
import { ConversationSidebar } from "@/components/conversation/conversation-sidebar"
import { ChatInterface } from "@/components/conversation/chat-interface"
import { AISteps } from "@/components/conversation/ai-steps"
import type { ProcessingStep } from "@/lib/api"

export default function ConversationPage() {
  const [showSteps, setShowSteps] = useState(false)
  const [currentSteps, setCurrentSteps] = useState<ProcessingStep[] | undefined>(undefined)
  const [responseTime, setResponseTime] = useState<number | undefined>(undefined)

  return (
    <MainLayout noPadding noScroll>
      <div style={{
        display: "flex",
        flex: 1,
        height: "100%",
        padding: "12px",
        gap: "var(--gap-island)",
        overflow: "hidden",
        background: "var(--color-bg)",
      }}>
        {/* Left: Chat list island */}
        <ConversationSidebar />

        {/* Center: Chat island */}
        <div style={{
          flex: 1,
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius-lg)",
          boxShadow: "var(--shadow-sm)",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}>
          <ChatInterface 
            onShowSteps={() => setShowSteps(!showSteps)} 
            onStepsUpdate={(steps, time) => {
              setCurrentSteps(steps)
              setResponseTime(time)
            }}
          />
        </div>

        {/* Right: AI steps island (conditionally shown) */}
        {showSteps && <AISteps steps={currentSteps} responseTime={responseTime} />}
      </div>
    </MainLayout>
  )
}
