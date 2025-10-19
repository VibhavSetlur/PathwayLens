'use client';

import { useMemo } from 'react';
import Plot from 'react-plotly.js';
import { Card, CardContent } from '@/components/ui/card';

interface PathwayData {
  pathway: string;
  pValue: number;
  genes: number;
  database: string;
}

interface DotPlotProps {
  data: PathwayData[];
}

export function DotPlot({ data }: DotPlotProps) {
  const plotData = useMemo(() => {
    const sortedData = [...data].sort((a, b) => a.pValue - b.pValue);
    
    const x = sortedData.map(d => -Math.log10(d.pValue));
    const y = sortedData.map(d => d.pathway);
    const sizes = sortedData.map(d => Math.max(d.genes * 2, 10));
    const colors = sortedData.map(d => {
      switch (d.database) {
        case 'KEGG': return '#3b82f6';
        case 'Reactome': return '#10b981';
        case 'GO': return '#f59e0b';
        default: return '#6b7280';
      }
    });
    const text = sortedData.map(d => 
      `Pathway: ${d.pathway}<br>` +
      `P-value: ${d.pValue.toExponential(2)}<br>` +
      `Genes: ${d.genes}<br>` +
      `Database: ${d.database}`
    );

    return [{
      x,
      y,
      mode: 'markers',
      type: 'scatter',
      marker: {
        size: sizes,
        color: colors,
        opacity: 0.7,
        line: {
          width: 1,
          color: 'white'
        }
      },
      text,
      hovertemplate: '%{text}<extra></extra>',
      name: 'Pathways'
    }];
  }, [data]);

  const layout = {
    title: {
      text: 'Pathway Enrichment Analysis',
      font: { size: 16 }
    },
    xaxis: {
      title: '-log10(P-value)',
      showgrid: true,
      gridcolor: '#e5e7eb'
    },
    yaxis: {
      title: 'Pathway',
      showgrid: true,
      gridcolor: '#e5e7eb'
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'Inter, sans-serif',
      size: 12
    },
    margin: { l: 200, r: 50, t: 50, b: 50 },
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
