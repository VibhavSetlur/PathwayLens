'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { ArrowRight, Play, Sparkles, Zap } from 'lucide-react';
import Link from 'next/link';

export function HeroSection() {
  return (
    <section className="relative overflow-hidden bg-gradient-to-br from-primary/5 via-background to-secondary/5">
      <div className="container mx-auto px-4 py-24 lg:py-32">
        <div className="grid gap-12 lg:grid-cols-2 lg:gap-16">
          {/* Content */}
          <div className="flex flex-col justify-center space-y-8">
            <div className="space-y-4">
              <div className="inline-flex items-center rounded-full border bg-background/50 px-3 py-1 text-sm font-medium backdrop-blur-sm">
                <Sparkles className="mr-2 h-4 w-4 text-primary" />
                Next-Generation Computational Biology Platform
              </div>
              
              <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
                Transform Your
                <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
                  {' '}Multi-Omics{' '}
                </span>
                Analysis
              </h1>
              
              <p className="text-xl text-muted-foreground lg:text-2xl">
                Advanced pathway analysis, interactive visualizations, and comprehensive 
                bioinformatics tools for genomics, transcriptomics, proteomics, and more.
              </p>
            </div>
            
            <div className="flex flex-col gap-4 sm:flex-row">
              <Button asChild size="lg" className="group">
                <Link href="/dashboard">
                  Get Started
                  <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                </Link>
              </Button>
              
              <Button variant="outline" size="lg" asChild>
                <Link href="/demo">
                  <Play className="mr-2 h-4 w-4" />
                  Watch Demo
                </Link>
              </Button>
            </div>
            
            <div className="flex items-center space-x-6 text-sm text-muted-foreground">
              <div className="flex items-center space-x-2">
                <Zap className="h-4 w-4 text-primary" />
                <span>Lightning Fast</span>
              </div>
              <div className="flex items-center space-x-2">
                <Sparkles className="h-4 w-4 text-primary" />
                <span>Interactive Visualizations</span>
              </div>
            </div>
          </div>
          
          {/* Visual */}
          <div className="flex items-center justify-center">
            <Card className="w-full max-w-lg">
              <CardContent className="p-8">
                <div className="space-y-6">
                  <div className="text-center">
                    <h3 className="text-2xl font-semibold">Pathway Analysis</h3>
                    <p className="text-muted-foreground">
                      Upload your data and get instant insights
                    </p>
                  </div>
                  
                  <div className="space-y-4">
                    <div className="flex items-center space-x-3">
                      <div className="h-2 w-2 rounded-full bg-primary" />
                      <span className="text-sm">Gene Expression Data</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="h-2 w-2 rounded-full bg-secondary" />
                      <span className="text-sm">Proteomics Data</span>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="h-2 w-2 rounded-full bg-success" />
                      <span className="text-sm">Metabolomics Data</span>
                    </div>
                  </div>
                  
                  <div className="rounded-lg bg-muted p-4">
                    <div className="space-y-2">
                      <div className="h-2 w-full rounded bg-primary/20" />
                      <div className="h-2 w-3/4 rounded bg-primary/20" />
                      <div className="h-2 w-1/2 rounded bg-primary/20" />
                    </div>
                    <p className="mt-2 text-center text-sm text-muted-foreground">
                      Processing your data...
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
      
      {/* Background decoration */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 h-80 w-80 rounded-full bg-secondary/10 blur-3xl" />
      </div>
    </section>
  );
}
