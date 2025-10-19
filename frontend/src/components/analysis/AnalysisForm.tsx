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
import { Loader2, Upload, FileText, Database, BarChart3, Dna, TestTube } from 'lucide-react';

interface AnalysisFormProps {
  onSubmit: (data: AnalysisFormData) => void;
  loading?: boolean;
  error?: string;
}

interface AnalysisFormData {
  analysisType: 'pathway_enrichment' | 'multi_omics' | 'statistical';
  method: string;
  inputData: string;
  parameters: Record<string, any>;
  outputFormat: string;
}

const AnalysisForm: React.FC<AnalysisFormProps> = ({ onSubmit, loading = false, error }) => {
  const [formData, setFormData] = useState<AnalysisFormData>({
    analysisType: 'pathway_enrichment',
    method: 'ORA',
    inputData: '',
    parameters: {},
    outputFormat: 'json'
  });

  const [inputType, setInputType] = useState<'text' | 'file'>('text');

  const analysisMethods = {
    pathway_enrichment: [
      { value: 'ORA', label: 'Over-Representation Analysis (ORA)' },
      { value: 'GSEA', label: 'Gene Set Enrichment Analysis (GSEA)' },
      { value: 'GSVA', label: 'Gene Set Variation Analysis (GSVA)' },
      { value: 'pathway_topology', label: 'Pathway Topology Analysis' },
      { value: 'consensus', label: 'Consensus Analysis' }
    ],
    multi_omics: [
      { value: 'data_integration', label: 'Data Integration' },
      { value: 'correlation_analysis', label: 'Correlation Analysis' },
      { value: 'network_analysis', label: 'Network Analysis' },
      { value: 'pathway_analysis', label: 'Pathway Analysis' },
      { value: 'biomarker_discovery', label: 'Biomarker Discovery' }
    ],
    statistical: [
      { value: 't_test', label: 'T-Test' },
      { value: 'anova', label: 'ANOVA' },
      { value: 'chi_square', label: 'Chi-Square Test' },
      { value: 'correlation', label: 'Correlation Analysis' },
      { value: 'regression', label: 'Regression Analysis' },
      { value: 'clustering', label: 'Clustering Analysis' },
      { value: 'pca', label: 'Principal Component Analysis' },
      { value: 'survival', label: 'Survival Analysis' }
    ]
  };

  const databases = [
    { value: 'KEGG', label: 'KEGG' },
    { value: 'Reactome', label: 'Reactome' },
    { value: 'GO', label: 'Gene Ontology' },
    { value: 'WikiPathways', label: 'WikiPathways' },
    { value: 'MSigDB', label: 'MSigDB' }
  ];

  const species = [
    { value: 'human', label: 'Human' },
    { value: 'mouse', label: 'Mouse' },
    { value: 'rat', label: 'Rat' },
    { value: 'yeast', label: 'Yeast' },
    { value: 'drosophila', label: 'Drosophila' }
  ];

  const omicsTypes = [
    { value: 'genomics', label: 'Genomics' },
    { value: 'transcriptomics', label: 'Transcriptomics' },
    { value: 'proteomics', label: 'Proteomics' },
    { value: 'metabolomics', label: 'Metabolomics' }
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

  const renderPathwayEnrichmentParams = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="database">Database</Label>
          <Select
            value={formData.parameters.database || 'KEGG'}
            onValueChange={(value) => handleParameterChange('database', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select database" />
            </SelectTrigger>
            <SelectContent>
              {databases.map(db => (
                <SelectItem key={db.value} value={db.value}>
                  {db.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="species">Species</Label>
          <Select
            value={formData.parameters.species || 'human'}
            onValueChange={(value) => handleParameterChange('species', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select species" />
            </SelectTrigger>
            <SelectContent>
              {species.map(sp => (
                <SelectItem key={sp.value} value={sp.value}>
                  {sp.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="threshold">Significance Threshold</Label>
          <Input
            id="threshold"
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={formData.parameters.threshold || 0.05}
            onChange={(e) => handleParameterChange('threshold', parseFloat(e.target.value))}
          />
        </div>
        <div>
          <Label htmlFor="minPathwaySize">Min Pathway Size</Label>
          <Input
            id="minPathwaySize"
            type="number"
            min="1"
            value={formData.parameters.minPathwaySize || 5}
            onChange={(e) => handleParameterChange('minPathwaySize', parseInt(e.target.value))}
          />
        </div>
      </div>
    </div>
  );

  const renderMultiOmicsParams = () => (
    <div className="space-y-4">
      <div>
        <Label htmlFor="omicsTypes">Omics Types</Label>
        <div className="flex flex-wrap gap-2 mt-2">
          {omicsTypes.map(omics => (
            <Badge
              key={omics.value}
              variant={formData.parameters.omicsTypes?.includes(omics.value) ? 'default' : 'outline'}
              className="cursor-pointer"
              onClick={() => {
                const currentTypes = formData.parameters.omicsTypes || [];
                const newTypes = currentTypes.includes(omics.value)
                  ? currentTypes.filter(t => t !== omics.value)
                  : [...currentTypes, omics.value];
                handleParameterChange('omicsTypes', newTypes);
              }}
            >
              {omics.label}
            </Badge>
          ))}
        </div>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="threshold">Significance Threshold</Label>
          <Input
            id="threshold"
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={formData.parameters.threshold || 0.05}
            onChange={(e) => handleParameterChange('threshold', parseFloat(e.target.value))}
          />
        </div>
        <div>
          <Label htmlFor="species">Species</Label>
          <Select
            value={formData.parameters.species || 'human'}
            onValueChange={(value) => handleParameterChange('species', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select species" />
            </SelectTrigger>
            <SelectContent>
              {species.map(sp => (
                <SelectItem key={sp.value} value={sp.value}>
                  {sp.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );

  const renderStatisticalParams = () => (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <Label htmlFor="testType">Test Type</Label>
          <Select
            value={formData.parameters.testType || 'two_tailed'}
            onValueChange={(value) => handleParameterChange('testType', value)}
          >
            <SelectTrigger>
              <SelectValue placeholder="Select test type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="one_tailed">One-Tailed</SelectItem>
              <SelectItem value="two_tailed">Two-Tailed</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div>
          <Label htmlFor="threshold">Significance Threshold</Label>
          <Input
            id="threshold"
            type="number"
            step="0.01"
            min="0"
            max="1"
            value={formData.parameters.threshold || 0.05}
            onChange={(e) => handleParameterChange('threshold', parseFloat(e.target.value))}
          />
        </div>
      </div>
    </div>
  );

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5" />
          Analysis Configuration
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Tabs value={formData.analysisType} onValueChange={(value) => handleInputChange('analysisType', value)}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="pathway_enrichment" className="flex items-center gap-2">
                <Database className="h-4 w-4" />
                Pathway Enrichment
              </TabsTrigger>
              <TabsTrigger value="multi_omics" className="flex items-center gap-2">
                <Dna className="h-4 w-4" />
                Multi-Omics
              </TabsTrigger>
              <TabsTrigger value="statistical" className="flex items-center gap-2">
                <TestTube className="h-4 w-4" />
                Statistical
              </TabsTrigger>
            </TabsList>

            <TabsContent value="pathway_enrichment" className="space-y-4">
              <div>
                <Label htmlFor="method">Analysis Method</Label>
                <Select
                  value={formData.method}
                  onValueChange={(value) => handleInputChange('method', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select method" />
                  </SelectTrigger>
                  <SelectContent>
                    {analysisMethods.pathway_enrichment.map(method => (
                      <SelectItem key={method.value} value={method.value}>
                        {method.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {renderPathwayEnrichmentParams()}
            </TabsContent>

            <TabsContent value="multi_omics" className="space-y-4">
              <div>
                <Label htmlFor="method">Analysis Method</Label>
                <Select
                  value={formData.method}
                  onValueChange={(value) => handleInputChange('method', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select method" />
                  </SelectTrigger>
                  <SelectContent>
                    {analysisMethods.multi_omics.map(method => (
                      <SelectItem key={method.value} value={method.value}>
                        {method.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {renderMultiOmicsParams()}
            </TabsContent>

            <TabsContent value="statistical" className="space-y-4">
              <div>
                <Label htmlFor="method">Analysis Method</Label>
                <Select
                  value={formData.method}
                  onValueChange={(value) => handleInputChange('method', value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select method" />
                  </SelectTrigger>
                  <SelectContent>
                    {analysisMethods.statistical.map(method => (
                      <SelectItem key={method.value} value={method.value}>
                        {method.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              {renderStatisticalParams()}
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
                  placeholder="Enter gene IDs, pathway IDs, or other data (one per line or comma-separated)"
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
                  <SelectItem value="json">JSON</SelectItem>
                  <SelectItem value="csv">CSV</SelectItem>
                  <SelectItem value="tsv">TSV</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <Button type="submit" disabled={loading} className="w-full">
            {loading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Running Analysis...
              </>
            ) : (
              'Start Analysis'
            )}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
};

export default AnalysisForm;
