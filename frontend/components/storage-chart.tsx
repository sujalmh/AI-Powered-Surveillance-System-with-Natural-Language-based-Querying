"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { date: "Apr 10", used: 80.2, added: 12.5 },
  { date: "Apr 11", used: 92.7, added: 14.3 },
  { date: "Apr 12", used: 107.0, added: 15.8 },
  { date: "Apr 13", used: 122.8, added: 16.2 },
  { date: "Apr 14", used: 139.0, added: 18.5 },
  { date: "Apr 15", used: 128.6, added: 14.8 }, // Some footage was deleted
]

export function StorageChart() {
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
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Total Used</span>
                        <span className="font-bold text-primary">{payload[0].payload.used} GB</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Added</span>
                        <span className="font-bold text-blue-500">{payload[0].payload.added} GB</span>
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
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
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
              tickFormatter={(value) => `${value} GB`}
            />
            <Area
              type="monotone"
              dataKey="used"
              stroke="hsl(var(--primary))"
              fill="hsl(var(--primary))"
              fillOpacity={0.2}
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="added"
              stroke="hsl(217, 91%, 60%)"
              fill="hsl(217, 91%, 60%)"
              fillOpacity={0.1}
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </Chart>
    </ChartContainer>
  )
}
