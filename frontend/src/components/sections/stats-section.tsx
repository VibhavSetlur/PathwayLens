'use client';

import { Card, CardContent } from '@/components/ui/card';
import { Database, FileText, Users, Zap } from 'lucide-react';

const stats = [
  {
    icon: Database,
    value: '8+',
    label: 'Pathway Databases',
    description: 'KEGG, Reactome, GO, BioCyc, and more',
  },
  {
    icon: FileText,
    value: '10+',
    label: 'Analysis Methods',
    description: 'ORA, GSEA, GSVA, topology analysis',
  },
  {
    icon: Users,
    value: '1000+',
    label: 'Active Researchers',
    description: 'Trusted by computational biologists worldwide',
  },
  {
    icon: Zap,
    value: '< 30s',
    label: 'Average Processing',
    description: 'Lightning-fast analysis and visualization',
  },
];

export function StatsSection() {
  return (
    <section className="py-24 lg:py-32 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat, index) => (
            <Card key={index} className="text-center">
              <CardContent className="pt-6">
                <div className="flex justify-center mb-4">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10">
                    <stat.icon className="h-8 w-8 text-primary" />
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="text-3xl font-bold text-primary">
                    {stat.value}
                  </div>
                  <div className="text-lg font-semibold">
                    {stat.label}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {stat.description}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
}
