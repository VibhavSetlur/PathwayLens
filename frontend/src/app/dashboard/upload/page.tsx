import { Metadata } from 'next';
import { DashboardLayout } from '@/components/layout/dashboard-layout';
import { FileUploadInterface } from '@/components/upload/file-upload-interface';

export const metadata: Metadata = {
  title: 'Upload Data - PathwayLens 2.0',
  description: 'Upload your multi-omics data for pathway analysis.',
};

export default function UploadPage() {
  return (
    <DashboardLayout>
      <div className="space-y-8">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Upload Data</h1>
          <p className="text-muted-foreground">
            Upload your gene expression, proteomics, or other omics data for analysis.
          </p>
        </div>
        <FileUploadInterface />
      </div>
    </DashboardLayout>
  );
}
