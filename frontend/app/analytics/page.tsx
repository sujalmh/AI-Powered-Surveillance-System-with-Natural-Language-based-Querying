import { MainLayout } from "@/components/layout/main-layout"
import { AnalyticsCharts } from "@/components/analytics/analytics-charts"

export default function AnalyticsPage() {
  return (
    <MainLayout>
      <div className="space-y-6">

        <AnalyticsCharts />
      </div>
    </MainLayout>
  )
}
