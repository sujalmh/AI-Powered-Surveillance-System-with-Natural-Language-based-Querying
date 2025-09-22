"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Line, LineChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { date: "Apr 10", logins: 12, newUsers: 2 },
  { date: "Apr 11", logins: 8, newUsers: 0 },
  { date: "Apr 12", logins: 15, newUsers: 1 },
  { date: "Apr 13", logins: 10, newUsers: 0 },
  { date: "Apr 14", logins: 18, newUsers: 3 },
  { date: "Apr 15", logins: 14, newUsers: 1 },
]

export function UsersChart() {
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
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Logins</span>
                        <span className="font-bold text-primary">{payload[0].payload.logins}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">New Users</span>
                        <span className="font-bold text-green-500">{payload[0].payload.newUsers}</span>
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
          <LineChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
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
            <Line
              type="monotone"
              dataKey="logins"
              stroke="hsl(var(--primary))"
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="newUsers"
              stroke="hsl(142, 76%, 36%)"
              strokeWidth={2}
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </Chart>
    </ChartContainer>
  )
}
