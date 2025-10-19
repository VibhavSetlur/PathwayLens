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

interface VolcanoPlotProps {
  data: GeneData[];
}

export function VolcanoPlot({ data }: VolcanoPlotProps) {
  const plotData = useMemo(() => {
    const x = data.map(d => d.logFC);
    const y = data.map(d => -Math.log10(d.pValue));
    const text = data.map(d => d.gene);
    const colors = data.map(d => {
      const isSignificant = d.pValue < 0.05 && Math.abs(d.logFC) > 1;
      return isSignificant ? '#ef4444' : '#6b7280';
    });
    const sizes = data.map(d => {
      const isSignificant = d.pValue < 0.05 && Math.abs(d.logFC) > 1;
      return isSignificant ? 8 : 4;
    });

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
      hovertemplate: 
        'Gene: %{text}<br>' +
        'Log2FC: %{x:.2f}<br>' +
        '-log10(P-value): %{y:.2f}<extra></extra>',
      name: 'Genes'
    }];
  }, [data]);

  const layout = {
    title: {
      text: 'Volcano Plot - Gene Expression Changes',
      font: { size: 16 }
    },
    xaxis: {
      title: 'Log2 Fold Change',
      showgrid: true,
      gridcolor: '#e5e7eb',
      zeroline: true,
      zerolinecolor: '#d1d5db'
    },
    yaxis: {
      title: '-log10(P-value)',
      showgrid: true,
      gridcolor: '#e5e7eb'
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'Inter, sans-serif',
      size: 12
    },
    margin: { l: 80, r: 50, t: 50, b: 60 },
    showlegend: false,
    shapes: [
      {
        type: 'line',
        x0: -1, x1: -1,
        y0: 0, y1: Math.max(...data.map(d => -Math.log10(d.pValue))),
        line: { color: '#d1d5db', width: 1, dash: 'dash' }
      },
      {
        type: 'line',
        x0: 1, x1: 1,
        y0: 0, y1: Math.max(...data.map(d => -Math.log10(d.pValue))),
        line: { color: '#d1d5db', width: 1, dash: 'dash' }
      },
      {
        type: 'line',
        x0: Math.min(...data.map(d => d.logFC)), x1: Math.max(...data.map(d => d.logFC)),
        y0: -Math.log10(0.05), y1: -Math.log10(0.05),
        line: { color: '#d1d5db', width: 1, dash: 'dash' }
      }
    ]
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
