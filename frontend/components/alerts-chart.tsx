"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { date: "Apr 10", critical: 1, high: 2, medium: 3, low: 2 },
  { date: "Apr 11", critical: 0, high: 1, medium: 2, low: 3 },
  { date: "Apr 12", critical: 0, high: 0, medium: 2, low: 1 },
  { date: "Apr 13", critical: 2, high: 1, medium: 0, low: 1 },
  { date: "Apr 14", critical: 0, high: 2, medium: 1, low: 0 },
  { date: "Apr 15", critical: 2, high: 1, medium: 1, low: 1 },
]

export function AlertsChart() {
  return (
    <ChartContainer
      className="h-[300px] w-full"
      tooltip={
        <ChartTooltip>
          <ChartTooltipContent
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="rounded-lg border bg-background p-2 shadow-sm">
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Date</span>
                        <span className="font-bold text-muted-foreground">{payload[0].payload.date}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Critical</span>
                        <span className="font-bold text-red-500">{payload[0].payload.critical}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">High</span>
                        <span className="font-bold text-orange-500">{payload[0].payload.high}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Medium</span>
                        <span className="font-bold text-yellow-500">{payload[0].payload.medium}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Low</span>
                        <span className="font-bold text-blue-500">{payload[0].payload.low}</span>
                      </div>
                    </div>
                  </div>
                )
              }
              return null
            }}
          />
        </ChartTooltip>
      }
    >
      <Chart>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis
              dataKey="date"
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="hsl(var(--muted-foreground))"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value}`}
            />
            <Bar dataKey="critical" stackId="a" fill="hsl(0, 84%, 60%)" radius={[0, 0, 0, 0]} />
            <Bar dataKey="high" stackId="a" fill="hsl(30, 84%, 60%)" radius={[0, 0, 0, 0]} />
            <Bar dataKey="medium" stackId="a" fill="hsl(60, 84%, 60%)" radius={[0, 0, 0, 0]} />
            <Bar dataKey="low" stackId="a" fill="hsl(210, 84%, 60%)" radius={[0, 0, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Chart>
    </ChartContainer>
  )
}
