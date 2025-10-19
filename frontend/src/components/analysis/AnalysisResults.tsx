import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  BarChart3, 
  Download, 
  Eye, 
  RefreshCw, 
  CheckCircle, 
  XCircle, 
  Clock,
  TrendingUp,
  Database,
  FileText
} from 'lucide-react';

interface AnalysisResultsProps {
  jobId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  results?: any;
  error?: string;
  onRefresh: () => void;
  onDownload: (format: string) => void;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({
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
              <Database className="h-4 w-4 text-blue-500" />
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
    </div>
  );

  const renderPathwayResults = () => {
    if (!results?.pathways) return <p>No pathway results available</p>;

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Pathway Enrichment Results</h3>
          <Badge variant="outline">{results.pathways.length} pathways</Badge>
        </div>
        
        <div className="overflow-x-auto">
          <table className="w-full border-collapse border border-gray-300">
            <thead>
              <tr className="bg-gray-50">
                <th className="border border-gray-300 px-4 py-2 text-left">Pathway</th>
                <th className="border border-gray-300 px-4 py-2 text-left">P-value</th>
                <th className="border border-gray-300 px-4 py-2 text-left">Adjusted P-value</th>
                <th className="border border-gray-300 px-4 py-2 text-left">Overlap</th>
                <th className="border border-gray-300 px-4 py-2 text-left">Size</th>
              </tr>
            </thead>
            <tbody>
              {results.pathways.map((pathway: any, index: number) => (
                <tr key={index} className="hover:bg-gray-50">
                  <td className="border border-gray-300 px-4 py-2">
                    <div>
                      <p className="font-medium">{pathway.name}</p>
                      <p className="text-sm text-gray-500">{pathway.id}</p>
                    </div>
                  </td>
                  <td className="border border-gray-300 px-4 py-2">
                    <Badge variant={pathway.p_value < 0.05 ? 'default' : 'outline'}>
                      {pathway.p_value?.toExponential(2) || 'N/A'}
                    </Badge>
                  </td>
                  <td className="border border-gray-300 px-4 py-2">
                    <Badge variant={pathway.adjusted_p_value < 0.05 ? 'default' : 'outline'}>
                      {pathway.adjusted_p_value?.toExponential(2) || 'N/A'}
                    </Badge>
                  </td>
                  <td className="border border-gray-300 px-4 py-2">
                    {pathway.overlap_count || pathway.overlap_genes?.length || 'N/A'}
                  </td>
                  <td className="border border-gray-300 px-4 py-2">
                    {pathway.pathway_size || 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  const renderMultiOmicsResults = () => {
    if (!results?.analysis_results) return <p>No multi-omics results available</p>;

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Multi-Omics Analysis Results</h3>
          <Badge variant="outline">{results.analysis_results.method}</Badge>
        </div>
        
        {results.analysis_results.summary && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(results.analysis_results.summary).map(([key, value]) => (
                  <div key={key} className="text-center">
                    <p className="text-2xl font-bold">{value}</p>
                    <p className="text-sm text-gray-500 capitalize">
                      {key.replace(/_/g, ' ')}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {results.analysis_results.data_integration && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Data Integration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <p><strong>Omics Types:</strong> {results.analysis_results.data_integration.omics_types?.join(', ')}</p>
                <p><strong>Samples:</strong> {results.analysis_results.data_integration.samples?.length}</p>
                <p><strong>Features:</strong> {results.analysis_results.data_integration.features ? Object.values(results.analysis_results.data_integration.features).flat().length : 'N/A'}</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    );
  };

  const renderStatisticalResults = () => {
    if (!results?.analysis_results) return <p>No statistical results available</p>;

    return (
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Statistical Analysis Results</h3>
          <Badge variant="outline">{results.analysis_results.method}</Badge>
        </div>
        
        {results.analysis_results.summary && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(results.analysis_results.summary).map(([key, value]) => (
                  <div key={key} className="text-center">
                    <p className="text-2xl font-bold">{value}</p>
                    <p className="text-sm text-gray-500 capitalize">
                      {key.replace(/_/g, ' ')}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {results.analysis_results.t_test_results && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">T-Test Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse border border-gray-300">
                  <thead>
                    <tr className="bg-gray-50">
                      <th className="border border-gray-300 px-4 py-2 text-left">Feature</th>
                      <th className="border border-gray-300 px-4 py-2 text-left">P-value</th>
                      <th className="border border-gray-300 px-4 py-2 text-left">Effect Size</th>
                      <th className="border border-gray-300 px-4 py-2 text-left">T-statistic</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.analysis_results.t_test_results.map((result: any, index: number) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="border border-gray-300 px-4 py-2 font-medium">
                          {result.feature}
                        </td>
                        <td className="border border-gray-300 px-4 py-2">
                          <Badge variant={result.p_value < 0.05 ? 'default' : 'outline'}>
                            {result.p_value?.toExponential(2) || 'N/A'}
                          </Badge>
                        </td>
                        <td className="border border-gray-300 px-4 py-2">
                          {result.effect_size?.toFixed(3) || 'N/A'}
                        </td>
                        <td className="border border-gray-300 px-4 py-2">
                          {result.t_statistic?.toFixed(3) || 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        )}
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
          <Button
            variant="outline"
            size="sm"
            onClick={() => onDownload('csv')}
            className="flex items-center gap-2"
          >
            <Download className="h-4 w-4" />
            CSV
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

  return (
    <Card className="w-full max-w-6xl mx-auto">
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Analysis Results
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
                onClick={() => onDownload('json')}
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
            <TabsTrigger value="pathways" disabled={!results?.pathways}>
              Pathways
            </TabsTrigger>
            <TabsTrigger value="multiomics" disabled={!results?.analysis_results}>
              Multi-Omics
            </TabsTrigger>
            <TabsTrigger value="raw">Raw Data</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="mt-4">
            {renderOverview()}
          </TabsContent>

          <TabsContent value="pathways" className="mt-4">
            {renderPathwayResults()}
          </TabsContent>

          <TabsContent value="multiomics" className="mt-4">
            {renderMultiOmicsResults()}
          </TabsContent>

          <TabsContent value="raw" className="mt-4">
            {renderRawResults()}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default AnalysisResults;
