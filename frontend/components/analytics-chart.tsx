"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { time: "00:00", people: 12, vehicles: 3, objects: 15 },
  { time: "02:00", people: 8, vehicles: 1, objects: 10 },
  { time: "04:00", people: 5, vehicles: 0, objects: 6 },
  { time: "06:00", people: 10, vehicles: 2, objects: 12 },
  { time: "08:00", people: 45, vehicles: 12, objects: 30 },
  { time: "10:00", people: 78, vehicles: 24, objects: 52 },
  { time: "12:00", people: 95, vehicles: 30, objects: 64 },
  { time: "14:00", people: 102, vehicles: 28, objects: 70 },
  { time: "16:00", people: 110, vehicles: 32, objects: 75 },
  { time: "18:00", people: 85, vehicles: 25, objects: 60 },
  { time: "20:00", people: 65, vehicles: 18, objects: 45 },
  { time: "22:00", people: 40, vehicles: 10, objects: 30 },
]

export function AnalyticsChart() {
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
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Time</span>
                        <span className="font-bold text-muted-foreground">{payload[0].payload.time}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">People</span>
                        <span className="font-bold">{payload[0].payload.people}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Vehicles</span>
                        <span className="font-bold">{payload[0].payload.vehicles}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Objects</span>
                        <span className="font-bold">{payload[0].payload.objects}</span>
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
              dataKey="time"
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
            <Area
              type="monotone"
              dataKey="people"
              stackId="1"
              stroke="hsl(var(--primary))"
              fill="hsl(var(--primary))"
              fillOpacity={0.6}
            />
            <Area
              type="monotone"
              dataKey="vehicles"
              stackId="2"
              stroke="hsl(217, 91%, 70%)"
              fill="hsl(217, 91%, 70%)"
              fillOpacity={0.6}
            />
            <Area
              type="monotone"
              dataKey="objects"
              stackId="3"
              stroke="hsl(217, 91%, 80%)"
              fill="hsl(217, 91%, 80%)"
              fillOpacity={0.6}
            />
          </AreaChart>
        </ResponsiveContainer>
      </Chart>
    </ChartContainer>
  )
}
