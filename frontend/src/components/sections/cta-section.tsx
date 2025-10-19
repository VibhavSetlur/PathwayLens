'use client';

import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { ArrowRight, Download, Github, Play } from 'lucide-react';
import Link from 'next/link';

export function CtaSection() {
  return (
    <section className="py-24 lg:py-32">
      <div className="container mx-auto px-4">
        <div className="mx-auto max-w-4xl">
          <Card className="relative overflow-hidden">
            <CardContent className="p-12 text-center">
              <div className="space-y-8">
                <div className="space-y-4">
                  <h2 className="text-3xl font-bold tracking-tight sm:text-4xl lg:text-5xl">
                    Ready to Transform Your Research?
                  </h2>
                  <p className="text-xl text-muted-foreground">
                    Join thousands of researchers using PathwayLens 2.0 for 
                    cutting-edge multi-omics pathway analysis.
                  </p>
                </div>
                
                <div className="flex flex-col gap-4 sm:flex-row sm:justify-center">
                  <Button asChild size="lg" className="group">
                    <Link href="/dashboard">
                      Start Analyzing
                      <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                    </Link>
                  </Button>
                  
                  <Button variant="outline" size="lg" asChild>
                    <Link href="/docs">
                      <Download className="mr-2 h-4 w-4" />
                      View Documentation
                    </Link>
                  </Button>
                </div>
                
                <div className="flex items-center justify-center space-x-6 text-sm text-muted-foreground">
                  <Link 
                    href="/demo" 
                    className="flex items-center space-x-2 hover:text-foreground transition-colors"
                  >
                    <Play className="h-4 w-4" />
                    <span>Watch Demo</span>
                  </Link>
                  <Link 
                    href="https://github.com/pathwaylens/pathwaylens" 
                    className="flex items-center space-x-2 hover:text-foreground transition-colors"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Github className="h-4 w-4" />
                    <span>View on GitHub</span>
                  </Link>
                </div>
              </div>
            </CardContent>
            
            {/* Background decoration */}
            <div className="absolute inset-0 -z-10 overflow-hidden">
              <div className="absolute -top-20 -right-20 h-40 w-40 rounded-full bg-primary/10 blur-3xl" />
              <div className="absolute -bottom-20 -left-20 h-40 w-40 rounded-full bg-secondary/10 blur-3xl" />
            </div>
          </Card>
        </div>
      </div>
    </section>
  );
}
