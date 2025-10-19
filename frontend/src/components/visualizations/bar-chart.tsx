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

interface BarChartProps {
  data: PathwayData[];
}

export function BarChart({ data }: BarChartProps) {
  const plotData = useMemo(() => {
    const sortedData = [...data].sort((a, b) => b.genes - a.genes);
    
    const x = sortedData.map(d => d.pathway);
    const y = sortedData.map(d => d.genes);
    const colors = sortedData.map(d => {
      switch (d.database) {
        case 'KEGG': return '#3b82f6';
        case 'Reactome': return '#10b981';
        case 'GO': return '#f59e0b';
        default: return '#6b7280';
      }
    });
    const text = sortedData.map(d => 
      `Genes: ${d.genes}<br>` +
      `P-value: ${d.pValue.toExponential(2)}<br>` +
      `Database: ${d.database}`
    );

    return [{
      x,
      y,
      type: 'bar',
      marker: {
        color: colors,
        opacity: 0.8,
        line: {
          width: 1,
          color: 'white'
        }
      },
      text: y.map(val => val.toString()),
      textposition: 'outside',
      hovertemplate: '%{text}<extra></extra>',
      name: 'Gene Count'
    }];
  }, [data]);

  const layout = {
    title: {
      text: 'Pathway Gene Counts',
      font: { size: 16 }
    },
    xaxis: {
      title: 'Pathway',
      showgrid: true,
      gridcolor: '#e5e7eb',
      tickangle: -45
    },
    yaxis: {
      title: 'Number of Genes',
      showgrid: true,
      gridcolor: '#e5e7eb'
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'Inter, sans-serif',
      size: 12
    },
    margin: { l: 80, r: 50, t: 50, b: 100 },
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
