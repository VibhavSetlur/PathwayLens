'use client';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Activity, 
  BarChart3, 
  Clock, 
  Database, 
  FileText, 
  Plus, 
  TrendingUp,
  Upload,
  Users
} from 'lucide-react';
import Link from 'next/link';

const stats = [
  {
    name: 'Total Projects',
    value: '12',
    change: '+2',
    changeType: 'positive',
    icon: Database,
  },
  {
    name: 'Active Analyses',
    value: '8',
    change: '+3',
    changeType: 'positive',
    icon: Activity,
  },
  {
    name: 'Completed Jobs',
    value: '156',
    change: '+12',
    changeType: 'positive',
    icon: BarChart3,
  },
  {
    name: 'Team Members',
    value: '4',
    change: '+1',
    changeType: 'positive',
    icon: Users,
  },
];

const recentJobs = [
  {
    id: '1',
    name: 'RNA-seq Analysis - Sample A',
    status: 'completed',
    progress: 100,
    createdAt: '2 hours ago',
    type: 'analysis',
  },
  {
    id: '2',
    name: 'Proteomics Data Normalization',
    status: 'running',
    progress: 65,
    createdAt: '1 hour ago',
    type: 'normalization',
  },
  {
    id: '3',
    name: 'Pathway Comparison Study',
    status: 'queued',
    progress: 0,
    createdAt: '30 minutes ago',
    type: 'comparison',
  },
  {
    id: '4',
    name: 'Metabolomics Visualization',
    status: 'failed',
    progress: 0,
    createdAt: '1 day ago',
    type: 'visualization',
  },
];

const quickActions = [
  {
    name: 'Upload Data',
    description: 'Upload gene expression or proteomics data',
    href: '/dashboard/upload',
    icon: Upload,
  },
  {
    name: 'New Analysis',
    description: 'Start a new pathway analysis',
    href: '/dashboard/analysis',
    icon: BarChart3,
  },
  {
    name: 'Create Project',
    description: 'Organize your analyses in projects',
    href: '/dashboard/projects/new',
    icon: Plus,
  },
  {
    name: 'View Reports',
    description: 'Access your analysis reports',
    href: '/dashboard/reports',
    icon: FileText,
  },
];

export function DashboardOverview() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Welcome back! Here's what's happening with your analyses.
          </p>
        </div>
        <Button asChild>
          <Link href="/dashboard/upload">
            <Upload className="mr-2 h-4 w-4" />
            Upload Data
          </Link>
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.name}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {stat.name}
              </CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className="text-xs text-muted-foreground">
                <span className={`inline-flex items-center ${
                  stat.changeType === 'positive' ? 'text-success' : 'text-destructive'
                }`}>
                  <TrendingUp className="mr-1 h-3 w-3" />
                  {stat.change}
                </span>
                {' '}from last month
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        {/* Recent Jobs */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Jobs</CardTitle>
            <CardDescription>
              Your latest analysis jobs and their status
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {recentJobs.map((job) => (
                <div key={job.id} className="flex items-center space-x-4">
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center justify-between">
                      <p className="text-sm font-medium">{job.name}</p>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        job.status === 'completed' ? 'bg-success/10 text-success' :
                        job.status === 'running' ? 'bg-primary/10 text-primary' :
                        job.status === 'queued' ? 'bg-warning/10 text-warning' :
                        'bg-destructive/10 text-destructive'
                      }`}>
                        {job.status}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span>{job.createdAt}</span>
                    </div>
                    {job.status === 'running' && (
                      <Progress value={job.progress} className="h-2" />
                    )}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4">
              <Button variant="outline" className="w-full" asChild>
                <Link href="/dashboard/jobs">View All Jobs</Link>
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>
              Get started with common tasks
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4">
              {quickActions.map((action) => (
                <Button
                  key={action.name}
                  variant="outline"
                  className="h-auto p-4 justify-start"
                  asChild
                >
                  <Link href={action.href}>
                    <action.icon className="mr-3 h-5 w-5" />
                    <div className="text-left">
                      <div className="font-medium">{action.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {action.description}
                      </div>
                    </div>
                  </Link>
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
