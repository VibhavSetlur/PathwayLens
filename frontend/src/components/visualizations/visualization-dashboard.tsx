'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { DotPlot } from './dot-plot';
import { VolcanoPlot } from './volcano-plot';
import { NetworkVisualization } from './network-visualization';
import { HeatmapVisualization } from './heatmap-visualization';
import { BarChart } from './bar-chart';
import { Download, Share, Settings } from 'lucide-react';

// Mock data for demonstrations
const mockPathwayData = [
  { pathway: 'Cell Cycle', pValue: 1.2e-5, genes: 45, database: 'KEGG' },
  { pathway: 'DNA Repair', pValue: 3.4e-4, genes: 32, database: 'Reactome' },
  { pathway: 'Apoptosis', pValue: 5.6e-3, genes: 28, database: 'GO' },
  { pathway: 'Immune Response', pValue: 7.8e-3, genes: 41, database: 'KEGG' },
  { pathway: 'Metabolism', pValue: 9.1e-3, genes: 67, database: 'Reactome' },
];

const mockGeneData = [
  { gene: 'TP53', logFC: 2.3, pValue: 1e-6, pathway: 'Cell Cycle' },
  { gene: 'BRCA1', logFC: 1.8, pValue: 2e-5, pathway: 'DNA Repair' },
  { gene: 'MYC', logFC: -1.5, pValue: 3e-4, pathway: 'Cell Cycle' },
  { gene: 'CDKN1A', logFC: 2.1, pValue: 4e-5, pathway: 'Cell Cycle' },
  { gene: 'ATM', logFC: 1.7, pValue: 5e-4, pathway: 'DNA Repair' },
];

export function VisualizationDashboard() {
  const [selectedVisualization, setSelectedVisualization] = useState('dot-plot');

  const visualizations = [
    { id: 'dot-plot', name: 'Dot Plot', description: 'Pathway enrichment results' },
    { id: 'volcano', name: 'Volcano Plot', description: 'Gene expression changes' },
    { id: 'network', name: 'Network', description: 'Pathway interaction network' },
    { id: 'heatmap', name: 'Heatmap', description: 'Gene expression patterns' },
    { id: 'bar-chart', name: 'Bar Chart', description: 'Pathway statistics' },
  ];

  const renderVisualization = () => {
    switch (selectedVisualization) {
      case 'dot-plot':
        return <DotPlot data={mockPathwayData} />;
      case 'volcano':
        return <VolcanoPlot data={mockGeneData} />;
      case 'network':
        return <NetworkVisualization data={mockPathwayData} />;
      case 'heatmap':
        return <HeatmapVisualization data={mockGeneData} />;
      case 'bar-chart':
        return <BarChart data={mockPathwayData} />;
      default:
        return <DotPlot data={mockPathwayData} />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Visualization Controls */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Interactive Visualizations</CardTitle>
              <CardDescription>
                Explore your pathway analysis results with interactive charts and plots
              </CardDescription>
            </div>
            <div className="flex space-x-2">
              <Button variant="outline" size="sm">
                <Settings className="mr-2 h-4 w-4" />
                Settings
              </Button>
              <Button variant="outline" size="sm">
                <Share className="mr-2 h-4 w-4" />
                Share
              </Button>
              <Button variant="outline" size="sm">
                <Download className="mr-2 h-4 w-4" />
                Export
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs value={selectedVisualization} onValueChange={setSelectedVisualization}>
            <TabsList className="grid w-full grid-cols-5">
              {visualizations.map((viz) => (
                <TabsTrigger key={viz.id} value={viz.id}>
                  {viz.name}
                </TabsTrigger>
              ))}
            </TabsList>
            
            <div className="mt-6">
              {visualizations.map((viz) => (
                <TabsContent key={viz.id} value={viz.id} className="space-y-4">
                  <div className="text-center">
                    <h3 className="text-lg font-semibold">{viz.name}</h3>
                    <p className="text-sm text-muted-foreground">{viz.description}</p>
                  </div>
                  <div className="h-[600px] w-full">
                    {renderVisualization()}
                  </div>
                </TabsContent>
              ))}
            </div>
          </Tabs>
        </CardContent>
      </Card>

      {/* Visualization Info */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Interactive Features</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm space-y-1 text-muted-foreground">
              <li>• Zoom and pan</li>
              <li>• Hover for details</li>
              <li>• Click to filter</li>
              <li>• Export as PNG/SVG</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Data Sources</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm space-y-1 text-muted-foreground">
              <li>• KEGG pathways</li>
              <li>• Reactome pathways</li>
              <li>• Gene Ontology</li>
              <li>• Custom gene sets</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Export Options</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="text-sm space-y-1 text-muted-foreground">
              <li>• High-resolution PNG</li>
              <li>• Vector SVG</li>
              <li>• Interactive HTML</li>
              <li>• Publication-ready PDF</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
