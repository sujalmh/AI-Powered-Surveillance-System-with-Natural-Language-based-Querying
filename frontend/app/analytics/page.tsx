import { MainLayout } from "@/components/layout/main-layout"
import { AnalyticsHeader } from "@/components/analytics/analytics-header"
import { AnalyticsCharts } from "@/components/analytics/analytics-charts"

export default function AnalyticsPage() {
  return (
    <MainLayout>
      <div className="space-y-6">
        <AnalyticsHeader />
        <AnalyticsCharts />
      </div>
    </MainLayout>
  )
}
