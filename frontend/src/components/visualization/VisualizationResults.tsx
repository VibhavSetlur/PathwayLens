import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  Eye, 
  Download, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  Clock,
  TrendingUp,
  BarChart3,
  FileText,
  ExternalLink
} from 'lucide-react';

interface VisualizationResultsProps {
  jobId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  results?: any;
  error?: string;
  onRefresh: () => void;
  onDownload: (format: string) => void;
}

const VisualizationResults: React.FC<VisualizationResultsProps> = ({
  jobId,
  status,
  progress,
  message,
  results,
  error,
  onRefresh,
  onDownload
}) => {
  const [activeTab, setActiveTab] = useState('overview');
  const [visualizationUrl, setVisualizationUrl] = useState<string | null>(null);

  useEffect(() => {
    if (results?.output_path && status === 'completed') {
      setVisualizationUrl(results.output_path);
    }
  }, [results, status]);

  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />;
      case 'running':
        return <RefreshCw className="h-5 w-5 text-blue-500 animate-spin" />;
      default:
        return <Clock className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  const renderOverview = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4 text-blue-500" />
              <span className="text-sm font-medium">Job ID</span>
            </div>
            <p className="text-lg font-semibold mt-1">{jobId}</p>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              {getStatusIcon()}
              <span className="text-sm font-medium">Status</span>
            </div>
            <Badge className={`mt-1 ${getStatusColor()}`}>
              {status.toUpperCase()}
            </Badge>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium">Progress</span>
            </div>
            <p className="text-lg font-semibold mt-1">{progress}%</p>
          </CardContent>
        </Card>
      </div>

      {message && (
        <Alert>
          <AlertDescription>{message}</AlertDescription>
        </Alert>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="w-full bg-gray-200 rounded-full h-2">
        <div 
          className="bg-blue-600 h-2 rounded-full transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      {status === 'completed' && visualizationUrl && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Visualization Ready</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex gap-2">
              <Button
                onClick={() => window.open(visualizationUrl, '_blank')}
                className="flex items-center gap-2"
              >
                <ExternalLink className="h-4 w-4" />
                Open Visualization
              </Button>
              <Button
                variant="outline"
                onClick={() => onDownload('html')}
                className="flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Download
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );

  const renderVisualization = () => {
    if (!visualizationUrl || status !== 'completed') {
      return (
        <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
          <div className="text-center">
            <Eye className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">Visualization not available</p>
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Visualization</h3>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.open(visualizationUrl, '_blank')}
              className="flex items-center gap-2"
            >
              <ExternalLink className="h-4 w-4" />
              Open in New Tab
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onDownload('html')}
              className="flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              Download
            </Button>
          </div>
        </div>
        
        <Card>
          <CardContent className="p-0">
            <iframe
              src={visualizationUrl}
              className="w-full h-96 border-0 rounded-lg"
              title="PathwayLens Visualization"
            />
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderDashboard = () => {
    if (!results?.dashboard_data || status !== 'completed') {
      return (
        <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
          <div className="text-center">
            <BarChart3 className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">Dashboard not available</p>
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Interactive Dashboard</h3>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => window.open(visualizationUrl, '_blank')}
              className="flex items-center gap-2"
            >
              <ExternalLink className="h-4 w-4" />
              Open in New Tab
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => onDownload('html')}
              className="flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              Download
            </Button>
          </div>
        </div>
        
        <Card>
          <CardContent className="p-0">
            <iframe
              src={visualizationUrl}
              className="w-full h-96 border-0 rounded-lg"
              title="PathwayLens Dashboard"
            />
          </CardContent>
        </Card>

        {results.dashboard_data.components && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {results.dashboard_data.components.map((component: any, index: number) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="text-sm">{component.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-xs text-gray-500">
                    Type: {component.type}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    );
  };

  const renderMetadata = () => {
    if (!results) return <p>No metadata available</p>;

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Visualization Metadata</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Parameters</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(results.parameters || {}).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="text-sm font-medium capitalize">
                      {key.replace(/_/g, ' ')}:
                    </span>
                    <span className="text-sm text-gray-600">
                      {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Output Information</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Output Path:</span>
                  <span className="text-sm text-gray-600">{results.output_path || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Format:</span>
                  <span className="text-sm text-gray-600">{results.format || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Size:</span>
                  <span className="text-sm text-gray-600">{results.file_size || 'N/A'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm font-medium">Created:</span>
                  <span className="text-sm text-gray-600">
                    {results.created_at ? new Date(results.created_at).toLocaleString() : 'N/A'}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  };

  const renderRawResults = () => (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Raw Results</h3>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onDownload('json')}
            className="flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            JSON
          </Button>
        </div>
      </div>
      
      <Card>
        <CardContent className="p-4">
          <pre className="text-sm overflow-auto max-h-96 bg-gray-50 p-4 rounded">
            {JSON.stringify(results, null, 2)}
          </pre>
        </CardContent>
      </Card>
    </div>
  );

  const getTabContent = () => {
    if (results?.visualization_type === 'dashboard') {
      return 'dashboard';
    }
    return 'visualization';
  };

  return (
    <Card className="w-full max-w-6xl mx-auto">
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Visualization Results
          </CardTitle>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onRefresh}
              className="flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
            {status === 'completed' && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => onDownload('html')}
                className="flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Download
              </Button>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="visualization" disabled={!visualizationUrl || status !== 'completed'}>
              Visualization
            </TabsTrigger>
            <TabsTrigger value="dashboard" disabled={!results?.dashboard_data || status !== 'completed'}>
              Dashboard
            </TabsTrigger>
            <TabsTrigger value="raw">Raw Data</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-4">
            {renderOverview()}
          </TabsContent>

          <TabsContent value="visualization" className="mt-4">
            {renderVisualization()}
          </TabsContent>

          <TabsContent value="dashboard" className="mt-4">
            {renderDashboard()}
          </TabsContent>

          <TabsContent value="raw" className="mt-4">
            {renderRawResults()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default VisualizationResults;
