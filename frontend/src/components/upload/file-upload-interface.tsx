'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  Upload, 
  FileText, 
  X, 
  CheckCircle, 
  AlertCircle,
  File,
  Database,
  BarChart3
} from 'lucide-react';
import { cn } from '@/lib/utils/cn';
import toast from 'react-hot-toast';

interface UploadedFile {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'completed' | 'error';
  progress: number;
  error?: string;
}

const supportedFormats = [
  { name: 'CSV', extensions: ['.csv'], description: 'Comma-separated values' },
  { name: 'TSV', extensions: ['.tsv'], description: 'Tab-separated values' },
  { name: 'Excel', extensions: ['.xlsx', '.xls'], description: 'Microsoft Excel' },
  { name: 'JSON', extensions: ['.json'], description: 'JavaScript Object Notation' },
];

const dataTypes = [
  {
    name: 'Gene Expression',
    description: 'RNA-seq, microarray, or other gene expression data',
    icon: BarChart3,
    formats: ['CSV', 'TSV', 'Excel'],
  },
  {
    name: 'Proteomics',
    description: 'Protein abundance or mass spectrometry data',
    icon: Database,
    formats: ['CSV', 'TSV', 'Excel'],
  },
  {
    name: 'Metabolomics',
    description: 'Metabolite concentration or abundance data',
    icon: Database,
    formats: ['CSV', 'TSV', 'Excel'],
  },
  {
    name: 'Gene List',
    description: 'Simple list of gene identifiers',
    icon: FileText,
    formats: ['CSV', 'TSV', 'TXT'],
  },
];

export function FileUploadInterface() {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [selectedDataType, setSelectedDataType] = useState<string>('');
  const [species, setSpecies] = useState<string>('human');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const newFiles: UploadedFile[] = acceptedFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'pending',
      progress: 0,
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);
    toast.success(`${acceptedFiles.length} file(s) added for upload`);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'text/tab-separated-values': ['.tsv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json'],
      'text/plain': ['.txt'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: true,
  });

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(file => file.id !== fileId));
  };

  const uploadFiles = async () => {
    if (uploadedFiles.length === 0) {
      toast.error('Please select files to upload');
      return;
    }

    if (!selectedDataType) {
      toast.error('Please select a data type');
      return;
    }

    // Update file statuses to uploading
    setUploadedFiles(prev => 
      prev.map(file => ({ ...file, status: 'uploading' as const }))
    );

    try {
      // Simulate upload progress
      for (const uploadedFile of uploadedFiles) {
        for (let progress = 0; progress <= 100; progress += 10) {
          await new Promise(resolve => setTimeout(resolve, 100));
          setUploadedFiles(prev => 
            prev.map(file => 
              file.id === uploadedFile.id 
                ? { ...file, progress }
                : file
            )
          );
        }

        // Mark as completed
        setUploadedFiles(prev => 
          prev.map(file => 
            file.id === uploadedFile.id 
              ? { ...file, status: 'completed' as const, progress: 100 }
              : file
          )
        );
      }

      toast.success('All files uploaded successfully!');
    } catch (error) {
      toast.error('Upload failed. Please try again.');
      setUploadedFiles(prev => 
        prev.map(file => ({ 
          ...file, 
          status: 'error' as const, 
          error: 'Upload failed' 
        }))
      );
    }
  };

  return (
    <div className="space-y-8">
      {/* Data Type Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Data Type</CardTitle>
          <CardDescription>
            Choose the type of data you're uploading to ensure proper processing
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {dataTypes.map((type) => (
              <Button
                key={type.name}
                variant={selectedDataType === type.name ? 'default' : 'outline'}
                className="h-auto p-4 flex flex-col items-start space-y-2"
                onClick={() => setSelectedDataType(type.name)}
              >
                <type.icon className="h-6 w-6" />
                <div className="text-left">
                  <div className="font-medium">{type.name}</div>
                  <div className="text-sm text-muted-foreground">
                    {type.description}
                  </div>
                </div>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Species Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Species</CardTitle>
          <CardDescription>
            Select the species for your data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-2 md:grid-cols-3">
            {['human', 'mouse', 'rat', 'drosophila', 'zebrafish'].map((spec) => (
              <Button
                key={spec}
                variant={species === spec ? 'default' : 'outline'}
                onClick={() => setSpecies(spec)}
                className="capitalize"
              >
                {spec}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* File Upload */}
      <Card>
        <CardHeader>
          <CardTitle>Upload Files</CardTitle>
          <CardDescription>
            Drag and drop your files here, or click to select files
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={cn(
              'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
              isDragActive 
                ? 'border-primary bg-primary/5' 
                : 'border-muted-foreground/25 hover:border-primary/50'
            )}
          >
            <input {...getInputProps()} />
            <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
            <div className="space-y-2">
              <p className="text-lg font-medium">
                {isDragActive ? 'Drop files here' : 'Upload your data files'}
              </p>
              <p className="text-sm text-muted-foreground">
                Supports CSV, TSV, Excel, and JSON formats (max 100MB)
              </p>
            </div>
          </div>

          {/* Supported Formats */}
          <div className="mt-6">
            <h4 className="text-sm font-medium mb-3">Supported Formats</h4>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {supportedFormats.map((format) => (
                <div key={format.name} className="flex items-center space-x-2 text-sm">
                  <File className="h-4 w-4 text-muted-foreground" />
                  <span className="font-medium">{format.name}</span>
                  <span className="text-muted-foreground">({format.description})</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Uploaded Files</CardTitle>
            <CardDescription>
              {uploadedFiles.length} file(s) ready for upload
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {uploadedFiles.map((uploadedFile) => (
                <div key={uploadedFile.id} className="flex items-center space-x-4 p-4 border rounded-lg">
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-5 w-5 text-muted-foreground" />
                        <span className="font-medium">{uploadedFile.file.name}</span>
                        <span className="text-sm text-muted-foreground">
                          ({(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB)
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        {uploadedFile.status === 'completed' && (
                          <CheckCircle className="h-5 w-5 text-success" />
                        )}
                        {uploadedFile.status === 'error' && (
                          <AlertCircle className="h-5 w-5 text-destructive" />
                        )}
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => removeFile(uploadedFile.id)}
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                    
                    {uploadedFile.status === 'uploading' && (
                      <Progress value={uploadedFile.progress} className="h-2" />
                    )}
                    
                    {uploadedFile.error && (
                      <p className="text-sm text-destructive">{uploadedFile.error}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 flex justify-end space-x-4">
              <Button variant="outline" onClick={() => setUploadedFiles([])}>
                Clear All
              </Button>
              <Button onClick={uploadFiles} disabled={uploadedFiles.some(f => f.status === 'uploading')}>
                Upload Files
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
