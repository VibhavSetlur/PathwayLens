import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  BarChart3, 
  Dna, 
  Eye, 
  Database, 
  TestTube, 
  TrendingUp,
  Activity,
  Network,
  FileText,
  Download,
  ArrowRight
} from 'lucide-react';
import Link from 'next/link';

export default function HomePage() {
  const features = [
    {
      icon: BarChart3,
      title: 'Pathway Enrichment Analysis',
      description: 'Perform comprehensive pathway enrichment analysis using ORA, GSEA, GSVA, and consensus methods.',
      href: '/analysis',
      color: 'text-blue-500'
    },
    {
      icon: Dna,
      title: 'Multi-Omics Integration',
      description: 'Integrate and analyze genomics, transcriptomics, proteomics, and metabolomics data.',
      href: '/analysis',
      color: 'text-green-500'
    },
    {
      icon: Eye,
      title: 'Interactive Visualizations',
      description: 'Create stunning visualizations and interactive dashboards for your analysis results.',
      href: '/visualization',
      color: 'text-purple-500'
    },
    {
      icon: Database,
      title: 'Data Normalization',
      description: 'Normalize gene IDs, pathway IDs, and omics data across different databases and species.',
      href: '/normalize',
      color: 'text-orange-500'
    },
    {
      icon: TestTube,
      title: 'Statistical Analysis',
      description: 'Perform comprehensive statistical analysis including t-tests, ANOVA, correlation, and more.',
      href: '/analysis',
      color: 'text-red-500'
    },
    {
      icon: TrendingUp,
      title: 'Comparison Tools',
      description: 'Compare analysis results, multi-omics data, and pathways across different conditions.',
      href: '/compare',
      color: 'text-indigo-500'
    }
  ];

  const analysisTypes = [
    {
      name: 'Over-Representation Analysis (ORA)',
      description: 'Identify over-represented pathways in your gene list',
      databases: ['KEGG', 'Reactome', 'GO', 'WikiPathways']
    },
    {
      name: 'Gene Set Enrichment Analysis (GSEA)',
      description: 'Analyze gene set enrichment using ranked gene lists',
      databases: ['KEGG', 'Reactome', 'GO', 'MSigDB']
    },
    {
      name: 'Gene Set Variation Analysis (GSVA)',
      description: 'Quantify pathway activity in individual samples',
      databases: ['KEGG', 'Reactome', 'GO', 'MSigDB']
    },
    {
      name: 'Consensus Analysis',
      description: 'Combine results from multiple analysis methods',
      databases: ['All supported databases']
    }
  ];

  const visualizationTypes = [
    {
      name: 'Bar Charts',
      description: 'Visualize pathway enrichment results',
      icon: BarChart3
    },
    {
      name: 'Volcano Plots',
      description: 'Display differential expression results',
      icon: TrendingUp
    },
    {
      name: 'Heatmaps',
      description: 'Show expression patterns across samples',
      icon: Activity
    },
    {
      name: 'Network Plots',
      description: 'Visualize pathway interactions',
      icon: Network
    },
    {
      name: 'Interactive Dashboards',
      description: 'Create comprehensive analysis dashboards',
      icon: Eye
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            PathwayLens
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-3xl mx-auto">
            A comprehensive pathway analysis platform for multi-omics data integration, 
            visualization, and statistical analysis
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/analysis">
              <Button size="lg" className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Start Analysis
              </Button>
            </Link>
            <Link href="/visualization">
              <Button size="lg" variant="outline" className="flex items-center gap-2">
                <Eye className="h-5 w-5" />
                Create Visualizations
              </Button>
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card key={index} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <Icon className={`h-6 w-6 ${feature.color}`} />
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600 mb-4">{feature.description}</p>
                  <Link href={feature.href}>
                    <Button variant="outline" size="sm" className="w-full">
                      Learn More
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            );
          })}
        </div>

        {/* Analysis Types Section */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-center mb-8">Analysis Methods</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {analysisTypes.map((analysis, index) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="text-lg">{analysis.name}</CardTitle>
                  <p className="text-gray-600">{analysis.description}</p>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-2">
                    {analysis.databases.map(db => (
                      <Badge key={db} variant="outline">{db}</Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Visualization Types Section */}
        <div className="mb-16">
          <h2 className="text-3xl font-bold text-center mb-8">Visualization Types</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {visualizationTypes.map((viz, index) => {
              const Icon = viz.icon;
              return (
                <Card key={index} className="text-center">
                  <CardContent className="pt-6">
                    <Icon className="h-12 w-12 mx-auto mb-4 text-blue-500" />
                    <h3 className="text-lg font-semibold mb-2">{viz.name}</h3>
                    <p className="text-gray-600">{viz.description}</p>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </div>

        {/* CTA Section */}
        <div className="text-center">
          <Card className="max-w-2xl mx-auto">
            <CardContent className="pt-8">
              <h2 className="text-2xl font-bold mb-4">Ready to Get Started?</h2>
              <p className="text-gray-600 mb-6">
                Upload your data and start analyzing pathways with PathwayLens today.
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Link href="/analysis">
                  <Button size="lg" className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Start Analysis
                  </Button>
                </Link>
                <Link href="/docs">
                  <Button size="lg" variant="outline" className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    View Documentation
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}