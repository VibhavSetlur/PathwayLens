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

interface NetworkVisualizationProps {
  data: PathwayData[];
}

export function NetworkVisualization({ data }: NetworkVisualizationProps) {
  const plotData = useMemo(() => {
    // Create nodes for pathways
    const nodes = data.map((pathway, index) => ({
      x: Math.cos((2 * Math.PI * index) / data.length),
      y: Math.sin((2 * Math.PI * index) / data.length),
      text: pathway.pathway,
      size: Math.max(pathway.genes * 0.5, 10),
      color: pathway.database === 'KEGG' ? '#3b82f6' : 
             pathway.database === 'Reactome' ? '#10b981' : '#f59e0b'
    }));

    // Create edges (simplified - in real app, would use actual pathway interactions)
    const edges = [];
    for (let i = 0; i < data.length; i++) {
      for (let j = i + 1; j < data.length; j++) {
        if (Math.random() > 0.7) { // Random connections for demo
          edges.push({
            x0: nodes[i].x,
            y0: nodes[i].y,
            x1: nodes[j].x,
            y1: nodes[j].y
          });
        }
      }
    }

    return [
      // Edges
      {
        x: edges.flatMap(edge => [edge.x0, edge.x1, null]),
        y: edges.flatMap(edge => [edge.y0, edge.y1, null]),
        mode: 'lines',
        line: { color: '#d1d5db', width: 1 },
        hoverinfo: 'none',
        showlegend: false
      },
      // Nodes
      {
        x: nodes.map(node => node.x),
        y: nodes.map(node => node.y),
        mode: 'markers+text',
        type: 'scatter',
        marker: {
          size: nodes.map(node => node.size),
          color: nodes.map(node => node.color),
          opacity: 0.8,
          line: { width: 2, color: 'white' }
        },
        text: nodes.map(node => node.text),
        textposition: 'middle center',
        textfont: { size: 10, color: 'white' },
        hovertemplate: 'Pathway: %{text}<br>Genes: %{marker.size}<extra></extra>',
        name: 'Pathways'
      }
    ];
  }, [data]);

  const layout = {
    title: {
      text: 'Pathway Interaction Network',
      font: { size: 16 }
    },
    xaxis: {
      showgrid: false,
      zeroline: false,
      showticklabels: false
    },
    yaxis: {
      showgrid: false,
      zeroline: false,
      showticklabels: false
    },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'Inter, sans-serif',
      size: 12
    },
    margin: { l: 50, r: 50, t: 50, b: 50 },
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
