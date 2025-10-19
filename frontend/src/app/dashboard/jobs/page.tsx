import { Metadata } from 'next';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { JobManagementInterface } from '@/components/jobs/job-management-interface';

export const metadata: Metadata = {
  title: 'Job Management - PathwayLens 2.0',
  description: 'Monitor and manage your analysis jobs.',
};

export default function JobsPage() {
  return (
    <DashboardLayout>
      <div className="space-y-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Job Management</h1>
          <p className="text-muted-foreground">
            Monitor the status of your analysis jobs and view results.
          </p>
        </div>
        <JobManagementInterface />
      </div>
    </DashboardLayout>
  );
}
