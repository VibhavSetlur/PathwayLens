'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { 
  Clock, 
  CheckCircle, 
  XCircle, 
  Play, 
  Pause, 
  Trash2, 
  Download,
  Eye,
  RefreshCw,
  Filter,
  Search
} from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { cn } from '@/lib/utils/cn';
import { formatDistanceToNow } from 'date-fns';

interface Job {
  id: string;
  name: string;
  type: 'normalization' | 'analysis' | 'comparison' | 'visualization';
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
  error?: string;
  parameters: Record<string, any>;
}

// Mock data for demonstration
const mockJobs: Job[] = [
  {
    id: '1',
    name: 'RNA-seq Analysis - Sample A',
    type: 'analysis',
    status: 'completed',
    progress: 100,
    createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
    startedAt: new Date(Date.now() - 2 * 60 * 60 * 1000 + 5 * 60 * 1000), // 5 minutes after creation
    completedAt: new Date(Date.now() - 1 * 60 * 60 * 1000), // 1 hour ago
    parameters: { species: 'human', databases: ['kegg', 'reactome'] }
  },
  {
    id: '2',
    name: 'Proteomics Data Normalization',
    type: 'normalization',
    status: 'running',
    progress: 65,
    createdAt: new Date(Date.now() - 1 * 60 * 60 * 1000), // 1 hour ago
    startedAt: new Date(Date.now() - 1 * 60 * 60 * 1000 + 2 * 60 * 1000), // 2 minutes after creation
    parameters: { species: 'mouse', targetType: 'symbol' }
  },
  {
    id: '3',
    name: 'Pathway Comparison Study',
    type: 'comparison',
    status: 'queued',
    progress: 0,
    createdAt: new Date(Date.now() - 30 * 60 * 1000), // 30 minutes ago
    parameters: { comparisonType: 'gene_overlap' }
  },
  {
    id: '4',
    name: 'Metabolomics Visualization',
    type: 'visualization',
    status: 'failed',
    progress: 0,
    createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000), // 1 day ago
    error: 'Invalid data format',
    parameters: { plotType: 'heatmap' }
  },
];

export function JobManagementInterface() {
  const [jobs, setJobs] = useState<Job[]>(mockJobs);
  const [filteredJobs, setFilteredJobs] = useState<Job[]>(mockJobs);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setJobs(prevJobs => 
        prevJobs.map(job => {
          if (job.status === 'running' && job.progress < 100) {
            const newProgress = Math.min(job.progress + Math.random() * 10, 100);
            return {
              ...job,
              progress: newProgress,
              status: newProgress === 100 ? 'completed' : job.status,
              completedAt: newProgress === 100 ? new Date() : job.completedAt
            };
          }
          return job;
        })
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Filter jobs based on search and filters
  useEffect(() => {
    let filtered = jobs;

    if (searchTerm) {
      filtered = filtered.filter(job =>
        job.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(job => job.status === statusFilter);
    }

    if (typeFilter !== 'all') {
      filtered = filtered.filter(job => job.type === typeFilter);
    }

    setFilteredJobs(filtered);
  }, [jobs, searchTerm, statusFilter, typeFilter]);

  const getStatusIcon = (status: Job['status']) => {
    switch (status) {
      case 'queued':
        return <Clock className="h-4 w-4" />;
      case 'running':
        return <Play className="h-4 w-4" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4" />;
      case 'failed':
        return <XCircle className="h-4 w-4" />;
      case 'cancelled':
        return <Pause className="h-4 w-4" />;
    }
  };

  const getStatusColor = (status: Job['status']) => {
    switch (status) {
      case 'queued':
        return 'bg-warning/10 text-warning';
      case 'running':
        return 'bg-primary/10 text-primary';
      case 'completed':
        return 'bg-success/10 text-success';
      case 'failed':
        return 'bg-destructive/10 text-destructive';
      case 'cancelled':
        return 'bg-muted/10 text-muted-foreground';
    }
  };

  const getTypeColor = (type: Job['type']) => {
    switch (type) {
      case 'normalization':
        return 'bg-blue-100 text-blue-800';
      case 'analysis':
        return 'bg-green-100 text-green-800';
      case 'comparison':
        return 'bg-purple-100 text-purple-800';
      case 'visualization':
        return 'bg-orange-100 text-orange-800';
    }
  };

  const handleCancelJob = (jobId: string) => {
    setJobs(prevJobs =>
      prevJobs.map(job =>
        job.id === jobId ? { ...job, status: 'cancelled' as const } : job
      )
    );
  };

  const handleDeleteJob = (jobId: string) => {
    setJobs(prevJobs => prevJobs.filter(job => job.id !== jobId));
  };

  const handleRefresh = () => {
    // In a real app, this would fetch fresh data from the API
    setJobs([...mockJobs]);
  };

  return (
    <div className="space-y-6">
      {/* Filters and Search */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Job Management</CardTitle>
              <CardDescription>
                Monitor and manage your analysis jobs
              </CardDescription>
            </div>
            <Button onClick={handleRefresh} variant="outline" size="sm">
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 sm:flex-row">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search jobs..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10"
              />
            </div>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-full sm:w-40">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="queued">Queued</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
                <SelectItem value="cancelled">Cancelled</SelectItem>
              </SelectContent>
            </Select>
            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-full sm:w-40">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="normalization">Normalization</SelectItem>
                <SelectItem value="analysis">Analysis</SelectItem>
                <SelectItem value="comparison">Comparison</SelectItem>
                <SelectItem value="visualization">Visualization</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Jobs List */}
      <div className="space-y-4">
        {filteredJobs.map((job) => (
          <Card key={job.id}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex-1 space-y-2">
                  <div className="flex items-center space-x-3">
                    <h3 className="font-semibold">{job.name}</h3>
                    <Badge className={cn('text-xs', getStatusColor(job.status))}>
                      {getStatusIcon(job.status)}
                      <span className="ml-1 capitalize">{job.status}</span>
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {job.type}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                    <span>Created {formatDistanceToNow(job.createdAt)} ago</span>
                    {job.startedAt && (
                      <span>Started {formatDistanceToNow(job.startedAt)} ago</span>
                    )}
                    {job.completedAt && (
                      <span>Completed {formatDistanceToNow(job.completedAt)} ago</span>
                    )}
                  </div>

                  {job.status === 'running' && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>Progress</span>
                        <span>{job.progress}%</span>
                      </div>
                      <Progress value={job.progress} className="h-2" />
                    </div>
                  )}

                  {job.error && (
                    <div className="text-sm text-destructive">
                      Error: {job.error}
                    </div>
                  )}
                </div>

                <div className="flex items-center space-x-2">
                  {job.status === 'completed' && (
                    <>
                      <Button variant="outline" size="sm">
                        <Eye className="mr-2 h-4 w-4" />
                        View
                      </Button>
                      <Button variant="outline" size="sm">
                        <Download className="mr-2 h-4 w-4" />
                        Download
                      </Button>
                    </>
                  )}
                  
                  {job.status === 'running' && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handleCancelJob(job.id)}
                    >
                      <Pause className="mr-2 h-4 w-4" />
                      Cancel
                    </Button>
                  )}
                  
                  {(job.status === 'failed' || job.status === 'cancelled') && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handleDeleteJob(job.id)}
                    >
                      <Trash2 className="mr-2 h-4 w-4" />
                      Delete
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}

        {filteredJobs.length === 0 && (
          <Card>
            <CardContent className="p-12 text-center">
              <div className="space-y-2">
                <h3 className="text-lg font-semibold">No jobs found</h3>
                <p className="text-muted-foreground">
                  {searchTerm || statusFilter !== 'all' || typeFilter !== 'all'
                    ? 'Try adjusting your filters to see more jobs.'
                    : 'You haven\'t created any analysis jobs yet.'}
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
