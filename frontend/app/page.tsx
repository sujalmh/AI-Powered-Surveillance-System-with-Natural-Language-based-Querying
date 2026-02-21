import { MainLayout } from "@/components/layout/main-layout"
import { SummaryCards } from "@/components/dashboard/summary-cards"
import { CameraGrid } from "@/components/dashboard/camera-grid"
import { RecentAlerts } from "@/components/dashboard/recent-alerts"

export default function DashboardPage() {
  return (
    <MainLayout>
      {/* Top stats row */}
      <SummaryCards />

      {/* Camera feed + recent alerts side by side */}
      <div style={{
        display: "flex",
        gap: "var(--gap-island)",
        flex: 1,
        minHeight: 0,
        alignItems: "flex-start",
      }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <CameraGrid />
        </div>
        <RecentAlerts />
      </div>
    </MainLayout>
  )
}
