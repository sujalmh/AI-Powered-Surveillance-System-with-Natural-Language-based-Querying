"use client"

import { Chart, ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, XAxis, YAxis } from "recharts"

const data = [
  { date: "Apr 10", loginSuccess: 24, loginFailure: 3, passwordReset: 1 },
  { date: "Apr 11", loginSuccess: 18, loginFailure: 2, passwordReset: 0 },
  { date: "Apr 12", loginSuccess: 22, loginFailure: 1, passwordReset: 2 },
  { date: "Apr 13", loginSuccess: 25, loginFailure: 4, passwordReset: 1 },
  { date: "Apr 14", loginSuccess: 30, loginFailure: 2, passwordReset: 3 },
  { date: "Apr 15", loginSuccess: 28, loginFailure: 5, passwordReset: 2 },
]

export function SecurityChart() {
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
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Successful Logins</span>
                        <span className="font-bold text-green-500">{payload[0].payload.loginSuccess}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Failed Logins</span>
                        <span className="font-bold text-red-500">{payload[0].payload.loginFailure}</span>
                      </div>
                      <div className="flex flex-col">
                        <span className="text-[0.70rem] uppercase text-muted-foreground">Password Resets</span>
                        <span className="font-bold text-blue-500">{payload[0].payload.passwordReset}</span>
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
            <Bar dataKey="loginSuccess" fill="hsl(142, 76%, 36%)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="loginFailure" fill="hsl(0, 84%, 60%)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="passwordReset" fill="hsl(217, 91%, 60%)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Chart>
    </ChartContainer>
  )
}
