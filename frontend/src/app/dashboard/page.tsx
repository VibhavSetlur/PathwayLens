import { Metadata } from 'next';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { DashboardOverview } from '@/components/dashboard/dashboard-overview';

export const metadata: Metadata = {
  title: 'Dashboard - PathwayLens 2.0',
  description: 'Manage your pathway analysis projects and view results.',
};

export default function DashboardPage() {
  return (
    <DashboardLayout>
      <DashboardOverview />
    </DashboardLayout>
  );
}
