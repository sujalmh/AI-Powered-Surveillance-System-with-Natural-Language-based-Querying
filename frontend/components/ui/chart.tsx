import * as React from "react"

import { cn } from "@/lib/utils"

const ChartContainer = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
  ({ className, ...props }, ref) => {
    return <div className={cn("relative", className)} ref={ref} {...props} />
  },
)
ChartContainer.displayName = "ChartContainer"

const Chart = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(({ className, ...props }, ref) => {
  return <div className={cn("relative", className)} ref={ref} {...props} />
})
Chart.displayName = "Chart"

interface ChartTooltipProps {
  children: React.ReactNode
}

const ChartTooltip = ({ children }: ChartTooltipProps) => {
  return <>{children}</>
}
ChartTooltip.displayName = "ChartTooltip"

interface ChartTooltipContentProps {
  content: React.ReactNode
}

const ChartTooltipContent = ({ content }: ChartTooltipContentProps) => {
  return <>{content}</>
}
ChartTooltipContent.displayName = "ChartTooltipContent"

export { Chart, ChartContainer, ChartTooltip, ChartTooltipContent }
