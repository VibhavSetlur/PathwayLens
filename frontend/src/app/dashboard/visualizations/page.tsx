import { Metadata } from 'next';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { VisualizationDashboard } from '@/components/visualizations/visualization-dashboard';

export const metadata: Metadata = {
  title: 'Visualizations - PathwayLens 2.0',
  description: 'Interactive visualizations for your pathway analysis results.',
};

export default function VisualizationsPage() {
  return (
    <DashboardLayout>
      <div className="space-y-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Visualizations</h1>
          <p className="text-muted-foreground">
            Explore your pathway analysis results with interactive visualizations.
          </p>
        </div>
        <VisualizationDashboard />
      </div>
    </DashboardLayout>
  );
}
