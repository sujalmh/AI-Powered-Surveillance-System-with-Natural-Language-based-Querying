import { MainLayout } from "@/components/layout/main-layout"
import { DashboardHeader } from "@/components/dashboard/dashboard-header"
import { SummaryCards } from "@/components/dashboard/summary-cards"
import { CameraGrid } from "@/components/dashboard/camera-grid"
import { RecentAlerts } from "@/components/dashboard/recent-alerts"

export default function DashboardPage() {
  return (
    <MainLayout>
      <div className="space-y-4">
        <DashboardHeader />
        <SummaryCards />
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2">
            <CameraGrid />
          </div>
          <div>
            <RecentAlerts />
          </div>
        </div>
      </div>
    </MainLayout>
  )
}
