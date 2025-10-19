import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Switch } from '@/components/ui/switch';
import { 
  Loader2, 
  Upload, 
  FileText, 
  BarChart3, 
  PieChart, 
  Scatter, 
  Network,
  Activity,
  TrendingUp,
  Eye,
  Download
} from 'lucide-react';

interface VisualizationFormProps {
  onSubmit: (data: VisualizationFormData) => void;
  loading?: boolean;
  error?: string;
}

interface VisualizationFormData {
  visualizationType: 'generate' | 'dashboard' | 'export';
  plotType: string;
  inputData: string;
  parameters: Record<string, any>;
  outputFormat: string;
}

const VisualizationForm: React.FC<VisualizationFormProps> = ({ onSubmit, loading = false, error }) => {
  const [formData, setFormData] = useState<VisualizationFormData>({
    visualizationType: 'generate',
    plotType: 'bar',
    inputData: '',
    parameters: {},
    outputFormat: 'html'
  });

  const [inputType, setInputType] = useState<'text' | 'file'>('text');

  const plotTypes = [
    { value: 'bar', label: 'Bar Chart', icon: BarChart3 },
    { value: 'scatter', label: 'Scatter Plot', icon: Scatter },
    { value: 'heatmap', label: 'Heatmap', icon: Activity },
    { value: 'network', label: 'Network Plot', icon: Network },
    { value: 'volcano', label: 'Volcano Plot', icon: TrendingUp },
    { value: 'manhattan', label: 'Manhattan Plot', icon: BarChart3 },
    { value: 'pathway_diagram', label: 'Pathway Diagram', icon: Network },
    { value: 'pie', label: 'Pie Chart', icon: PieChart }
  ];

  const dashboardTypes = [
    { value: 'multi_omics', label: 'Multi-Omics Dashboard' },
    { value: 'pathway_analysis', label: 'Pathway Analysis Dashboard' },
    { value: 'comparison', label: 'Comparison Dashboard' },
    { value: 'custom', label: 'Custom Dashboard' }
  ];

  const layouts = [
    { value: 'grid', label: 'Grid Layout' },
    { value: 'tabbed', label: 'Tabbed Layout' },
    { value: 'sidebar', label: 'Sidebar Layout' }
  ];

  const colorSchemes = [
    { value: 'viridis', label: 'Viridis' },
    { value: 'plasma', label: 'Plasma' },
    { value: 'inferno', label: 'Inferno' },
    { value: 'magma', label: 'Magma' },
    { value: 'cividis', label: 'Cividis' },
    { value: 'blues', label: 'Blues' },
    { value: 'reds', label: 'Reds' },
    { value: 'greens', label: 'Greens' }
  ];

  const handleInputChange = (field: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleParameterChange = (key: string, value: any) => {
    setFormData(prev => ({
      ...prev,
      parameters: {
        ...prev.parameters,
        [key]: value
      }
    }));
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        setFormData(prev => ({
          ...prev,
          inputData: content
        }));
      };
      reader.readAsText(file);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  const renderGenerateParams = () => (
    <div className="space-y-4">
      <div>
        <Label htmlFor="plotType">Plot Type</Label>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
          {plotTypes.map(plot => {
            const Icon = plot.icon;
            return (
              <Button
                key={plot.value}
                type="button"
                variant={formData.plotType === plot.value ? 'default' : 'outline'}
                onClick={() => handleInputChange('plotType', plot.value)}
                className="flex flex-col items-center gap-2 h-auto p-3"
              >
                <Icon className="h-4 w-4" />
                <span className="text-xs">{plot.label}</span>
              </Button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="width">Width</Label>
          <Input
            id="width"
            type="number"
            min="100"
            max="2000"
            value={formData.parameters.width || 800}
            onChange={(e) => handleParameterChange('width', parseInt(e.target.value))}
          />
        </div>
        <div>
          <Label htmlFor="height">Height</Label>
          <Input
            id="height"
            type="number"
            min="100"
            max="2000"
            value={formData.parameters.height || 600}
            onChange={(e) => handleParameterChange('height', parseInt(e.target.value))}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="title">Title</Label>
          <Input
            id="title"
            value={formData.parameters.title || ''}
            onChange={(e) => handleParameterChange('title', e.target.value)}
            placeholder="Enter plot title"
          />
        </div>
        <div>
          <Label htmlFor="colorScheme">Color Scheme</Label>
          <Select
            value={formData.parameters.colorScheme || 'viridis'}
            onValueChange={(value) => handleParameterChange('colorScheme', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select color scheme" />
            </SelectTrigger>
            <SelectContent>
              {colorSchemes.map(scheme => (
                <SelectItem key={scheme.value} value={scheme.value}>
                  {scheme.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <Switch
          id="interactive"
          checked={formData.parameters.interactive !== false}
          onCheckedChange={(checked) => handleParameterChange('interactive', checked)}
        />
        <Label htmlFor="interactive">Interactive Plot</Label>
      </div>
    </div>
  );

  const renderDashboardParams = () => (
    <div className="space-y-4">
      <div>
        <Label htmlFor="dashboardType">Dashboard Type</Label>
        <Select
          value={formData.parameters.dashboardType || 'multi_omics'}
          onValueChange={(value) => handleParameterChange('dashboardType', value)}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select dashboard type" />
          </SelectTrigger>
          <SelectContent>
            {dashboardTypes.map(type => (
              <SelectItem key={type.value} value={type.value}>
                {type.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="layout">Layout</Label>
          <Select
            value={formData.parameters.layout || 'grid'}
            onValueChange={(value) => handleParameterChange('layout', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select layout" />
            </SelectTrigger>
            <SelectContent>
              {layouts.map(layout => (
                <SelectItem key={layout.value} value={layout.value}>
                  {layout.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="title">Title</Label>
          <Input
            id="title"
            value={formData.parameters.title || ''}
            onChange={(e) => handleParameterChange('title', e.target.value)}
            placeholder="Enter dashboard title"
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="width">Width</Label>
          <Input
            id="width"
            type="number"
            min="100"
            max="2000"
            value={formData.parameters.width || 1200}
            onChange={(e) => handleParameterChange('width', parseInt(e.target.value))}
          />
        </div>
        <div>
          <Label htmlFor="height">Height</Label>
          <Input
            id="height"
            type="number"
            min="100"
            max="2000"
            value={formData.parameters.height || 800}
            onChange={(e) => handleParameterChange('height', parseInt(e.target.value))}
          />
        </div>
      </div>

      <div className="flex items-center space-x-2">
        <Switch
          id="interactive"
          checked={formData.parameters.interactive !== false}
          onCheckedChange={(checked) => handleParameterChange('interactive', checked)}
        />
        <Label htmlFor="interactive">Interactive Dashboard</Label>
      </div>

      <div className="flex items-center space-x-2">
        <Switch
          id="responsive"
          checked={formData.parameters.responsive !== false}
          onCheckedChange={(checked) => handleParameterChange('responsive', checked)}
        />
        <Label htmlFor="responsive">Responsive Design</Label>
      </div>
    </div>
  );

  const renderExportParams = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="exportFormat">Export Format</Label>
          <Select
            value={formData.parameters.exportFormat || 'png'}
            onValueChange={(value) => handleParameterChange('exportFormat', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select export format" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="png">PNG</SelectItem>
              <SelectItem value="svg">SVG</SelectItem>
              <SelectItem value="pdf">PDF</SelectItem>
              <SelectItem value="jpg">JPG</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="dpi">DPI</Label>
          <Input
            id="dpi"
            type="number"
            min="72"
            max="600"
            value={formData.parameters.dpi || 300}
            onChange={(e) => handleParameterChange('dpi', parseInt(e.target.value))}
          />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="width">Width</Label>
          <Input
            id="width"
            type="number"
            min="100"
            max="2000"
            value={formData.parameters.width || 800}
            onChange={(e) => handleParameterChange('width', parseInt(e.target.value))}
          />
        </div>
        <div>
          <Label htmlFor="height">Height</Label>
          <Input
            id="height"
            type="number"
            min="100"
            max="2000"
            value={formData.parameters.height || 600}
            onChange={(e) => handleParameterChange('height', parseInt(e.target.value))}
          />
        </div>
      </div>
    </div>
  );

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Eye className="h-5 w-5" />
          Visualization Configuration
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Tabs value={formData.visualizationType} onValueChange={(value) => handleInputChange('visualizationType', value)}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="generate" className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Generate Plot
              </TabsTrigger>
              <TabsTrigger value="dashboard" className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Create Dashboard
              </TabsTrigger>
              <TabsTrigger value="export" className="flex items-center gap-2">
                <Download className="h-4 w-4" />
                Export
              </TabsTrigger>
            </TabsList>

            <TabsContent value="generate" className="space-y-4">
              {renderGenerateParams()}
            </TabsContent>

            <TabsContent value="dashboard" className="space-y-4">
              {renderDashboardParams()}
            </TabsContent>

            <TabsContent value="export" className="space-y-4">
              {renderExportParams()}
            </TabsContent>
          </Tabs>

          <div className="space-y-4">
            <div>
              <Label htmlFor="inputType">Input Type</Label>
              <div className="flex gap-4 mt-2">
                <Button
                  type="button"
                  variant={inputType === 'text' ? 'default' : 'outline'}
                  onClick={() => setInputType('text')}
                  className="flex items-center gap-2"
                >
                  <FileText className="h-4 w-4" />
                  Text Input
                </Button>
                <Button
                  type="button"
                  variant={inputType === 'file' ? 'default' : 'outline'}
                  onClick={() => setInputType('file')}
                  className="flex items-center gap-2"
                >
                  <Upload className="h-4 w-4" />
                  File Upload
                </Button>
              </div>
            </div>

            {inputType === 'text' ? (
              <div>
                <Label htmlFor="inputData">Input Data</Label>
                <Textarea
                  id="inputData"
                  placeholder="Enter data for visualization (JSON, CSV, or other format)"
                  value={formData.inputData}
                  onChange={(e) => handleInputChange('inputData', e.target.value)}
                  rows={6}
                />
              </div>
            ) : (
              <div>
                <Label htmlFor="fileUpload">Upload File</Label>
                <Input
                  id="fileUpload"
                  type="file"
                  accept=".txt,.csv,.tsv,.json,.yaml,.yml"
                  onChange={handleFileUpload}
                />
              </div>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="outputFormat">Output Format</Label>
              <Select
                value={formData.outputFormat}
                onValueChange={(value) => handleInputChange('outputFormat', value)}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select output format" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="html">HTML</SelectItem>
                  <SelectItem value="png">PNG</SelectItem>
                  <SelectItem value="svg">SVG</SelectItem>
                  <SelectItem value="pdf">PDF</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button type="submit" disabled={loading} className="w-full">
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating Visualization...
              </>
            ) : (
              'Generate Visualization'
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default VisualizationForm;
