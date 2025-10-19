'use client';

import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Card, CardContent } from '@/components/ui/card';

interface GeneData {
  gene: string;
  logFC: number;
  pValue: number;
  pathway: string;
}

interface HeatmapVisualizationProps {
  data: GeneData[];
}

export function HeatmapVisualization({ data }: HeatmapVisualizationProps) {
  const plotData = useMemo(() => {
    // Create mock expression data for multiple samples
    const samples = ['Sample A', 'Sample B', 'Sample C', 'Sample D', 'Sample E'];
    const genes = data.map(d => d.gene);
    
    // Generate random expression values for demo
    const z = genes.map(() => 
      samples.map(() => Math.random() * 4 - 2) // Values between -2 and 2
    );

    return [{
      z,
      x: samples,
      y: genes,
      type: 'heatmap',
      colorscale: [
        [0, '#3b82f6'],
        [0.5, '#ffffff'],
        [1, '#ef4444']
      ],
      hovertemplate: 
        'Gene: %{y}<br>' +
        'Sample: %{x}<br>' +
        'Expression: %{z:.2f}<extra></extra>',
      name: 'Expression'
    }];
  }, [data]);

  const layout = {
    title: {
      text: 'Gene Expression Heatmap',
      font: { size: 16 }
    },
    xaxis: {
      title: 'Samples',
      showgrid: true,
      gridcolor: '#e5e7eb'
    },
    yaxis: {
      title: 'Genes',
      showgrid: true,
      gridcolor: '#e5e7eb'
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'Inter, sans-serif',
      size: 12
    },
    margin: { l: 100, r: 50, t: 50, b: 60 },
    showlegend: false
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    responsive: true
  };

  return (
    <Card className="h-full">
      <CardContent className="p-0 h-full">
        <Plot
          data={plotData}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '100%' }}
        />
      </CardContent>
    </Card>
  );
}
