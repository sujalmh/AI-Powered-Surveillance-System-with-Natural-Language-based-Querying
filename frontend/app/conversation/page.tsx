"use client"

import { MainLayout } from "@/components/layout/main-layout"
import { ConversationSidebar } from "@/components/conversation/conversation-sidebar"
import { ChatInterface } from "@/components/conversation/chat-interface"
import { AISteps } from "@/components/conversation/ai-steps"
import { useState } from "react"

export default function ConversationPage() {
  const [showSteps, setShowSteps] = useState(false)

  return (
    <MainLayout>
      <div className="flex gap-6 h-full">
        <ConversationSidebar />
        <ChatInterface onShowSteps={() => setShowSteps(!showSteps)} />
        {showSteps && <AISteps />}
      </div>
    </MainLayout>
  )
}
