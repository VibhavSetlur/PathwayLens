'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  BarChart3, 
  Database, 
  FileText, 
  GitBranch, 
  Globe, 
  Layers, 
  Network, 
  Zap 
} from 'lucide-react';

const features = [
  {
    icon: Layers,
    title: 'Multi-Omics Support',
    description: 'Comprehensive analysis across genomics, transcriptomics, proteomics, metabolomics, phosphoproteomics, and epigenomics.',
  },
  {
    icon: Network,
    title: 'Advanced Pathway Analysis',
    description: 'ORA, GSEA, GSVA, and pathway topology methods with support for KEGG, Reactome, GO, and more.',
  },
  {
    icon: BarChart3,
    title: 'Interactive Visualizations',
    description: 'Publication-ready plots, interactive dashboards, and network visualizations with real-time filtering.',
  },
  {
    icon: GitBranch,
    title: 'Cross-Species Analysis',
    description: 'Seamless gene ID conversion and ortholog mapping across human, mouse, and model organisms.',
  },
  {
    icon: Database,
    title: 'Comprehensive Databases',
    description: 'Integration with KEGG, Reactome, Gene Ontology, BioCyc, Pathway Commons, and custom gene sets.',
  },
  {
    icon: FileText,
    title: 'Reproducible Workflows',
    description: 'Job tracking, configuration files, versioning, and automated report generation.',
  },
  {
    icon: Globe,
    title: 'Dual Interface',
    description: 'Zero-code web UI for researchers and standalone CLI for power users and automation.',
  },
  {
    icon: Zap,
    title: 'High Performance',
    description: 'Async processing, job queues, caching, and scalable architecture for large datasets.',
  },
];

export function FeaturesSection() {
  return (
    <section className="py-24 lg:py-32">
      <div className="container mx-auto px-4">
        <div className="mx-auto max-w-3xl text-center">
          <h2 className="text-3xl font-bold tracking-tight sm:text-4xl lg:text-5xl">
            Powerful Features for Modern Bioinformatics
          </h2>
          <p className="mt-6 text-lg text-muted-foreground">
            Everything you need for comprehensive multi-omics pathway analysis, 
            from data normalization to publication-ready visualizations.
          </p>
        </div>
        
        <div className="mt-16 grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((feature, index) => (
            <Card key={index} className="group hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <CardTitle className="text-xl">{feature.title}</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription className="text-base">
                  {feature.description}
                </CardDescription>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
