# PathwayLens 2.0 API Reference

## Overview

PathwayLens 2.0 provides a RESTful API for programmatic access to pathway analysis functionality. The API is built with FastAPI and provides comprehensive endpoints for data processing, analysis, and visualization.

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

PathwayLens 2.0 uses JWT-based authentication. Include the access token in the Authorization header:

```
Authorization: Bearer <access_token>
```

## Rate Limiting

API requests are rate-limited to prevent abuse:

- **Authenticated users**: 1000 requests per hour
- **Anonymous users**: 100 requests per hour

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## Error Handling

The API uses standard HTTP status codes and returns error details in JSON format:

```json
{
  "error": "validation_error",
  "message": "Invalid input data",
  "details": {
    "field": "species",
    "issue": "Invalid species code"
  }
}
```

## Endpoints

### Authentication

#### POST /auth/login

Authenticate user and get access token.

**Request Body**:
```json
{
  "username": "user@example.com",
  "password": "password123"
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### POST /auth/refresh

Refresh access token.

**Request Body**:
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Data Normalization

#### POST /normalize

Normalize gene identifiers across formats and species.

**Request Body**:
```json
{
  "data": [
    {"gene_id": "TP53", "value": 2.5},
    {"gene_id": "BRCA1", "value": 1.8}
  ],
  "species": "human",
  "target_species": "mouse",
  "target_type": "ensembl",
  "ambiguity_policy": "expand"
}
```

**Response**:
```json
{
  "normalized_data": [
    {
      "original_id": "TP53",
      "normalized_id": "ENSMUSG00000059552",
      "value": 2.5,
      "confidence": 0.95
    }
  ],
  "conversion_stats": {
    "total_genes": 2,
    "converted_genes": 2,
    "conversion_rate": 1.0
  }
}
```

#### POST /normalize/batch

Normalize multiple datasets in batch.

**Request Body**:
```json
{
  "datasets": [
    {
      "name": "dataset1",
      "data": [{"gene_id": "TP53", "value": 2.5}]
    },
    {
      "name": "dataset2", 
      "data": [{"gene_id": "BRCA1", "value": 1.8}]
    }
  ],
  "species": "human",
  "target_type": "ensembl"
}
```

### Pathway Analysis

#### POST /analyze/ora

Perform Over-Representation Analysis (ORA).

**Request Body**:
```json
{
  "genes": ["TP53", "BRCA1", "MYC"],
  "species": "human",
  "databases": ["kegg", "reactome", "go"],
  "parameters": {
    "significance_threshold": 0.05,
    "correction_method": "fdr_bh",
    "min_pathway_size": 5,
    "max_pathway_size": 500
  }
}
```

**Response**:
```json
{
  "analysis_id": "ora_12345",
  "status": "completed",
  "results": {
    "pathways": [
      {
        "pathway_id": "hsa04115",
        "pathway_name": "p53 signaling pathway",
        "database": "kegg",
        "p_value": 0.001,
        "adjusted_p_value": 0.01,
        "genes": ["TP53", "MDM2", "CDKN1A"],
        "gene_ratio": "3/10",
        "background_ratio": "50/20000"
      }
    ],
    "summary": {
      "total_pathways": 150,
      "significant_pathways": 25,
      "databases_used": ["kegg", "reactome", "go"]
    }
  }
}
```

#### POST /analyze/gsea

Perform Gene Set Enrichment Analysis (GSEA).

**Request Body**:
```json
{
  "ranked_genes": [
    {"gene_id": "TP53", "rank_score": 2.5},
    {"gene_id": "BRCA1", "rank_score": 1.8}
  ],
  "species": "human",
  "databases": ["kegg", "reactome"],
  "parameters": {
    "significance_threshold": 0.05,
    "min_pathway_size": 5,
    "max_pathway_size": 500
  }
}
```

**Response**:
```json
{
  "analysis_id": "gsea_12345",
  "status": "completed",
  "results": {
    "pathways": [
      {
        "pathway_id": "hsa04115",
        "pathway_name": "p53 signaling pathway",
        "database": "kegg",
        "enrichment_score": 0.75,
        "normalized_enrichment_score": 2.1,
        "p_value": 0.001,
        "adjusted_p_value": 0.01,
        "leading_edge": {
          "genes": ["TP53", "MDM2"],
          "fraction": 0.8
        }
      }
    ],
    "summary": {
      "total_pathways": 150,
      "significant_pathways": 25,
      "databases_used": ["kegg", "reactome"]
    }
  }
}
```

### Dataset Comparison

#### POST /compare

Compare multiple datasets or analysis results.

**Request Body**:
```json
{
  "comparison_type": "pathway_concordance",
  "datasets": [
    {
      "name": "condition1",
      "analysis_id": "ora_12345"
    },
    {
      "name": "condition2",
      "analysis_id": "ora_12346"
    }
  ],
  "parameters": {
    "significance_threshold": 0.05,
    "min_overlap": 2
  }
}
```

**Response**:
```json
{
  "comparison_id": "comp_12345",
  "status": "completed",
  "results": {
    "comparisons": [
      {
        "pathway_id": "hsa04115",
        "pathway_name": "p53 signaling pathway",
        "condition1": {
          "p_value": 0.001,
          "genes": ["TP53", "MDM2"]
        },
        "condition2": {
          "p_value": 0.01,
          "genes": ["TP53", "CDKN1A"]
        },
        "overlap": {
          "genes": ["TP53"],
          "jaccard_index": 0.33
        }
      }
    ],
    "summary": {
      "total_pathways": 150,
      "overlapping_pathways": 45,
      "unique_to_condition1": 30,
      "unique_to_condition2": 25
    }
  }
}
```

### Visualization

#### POST /visualize

Generate visualizations from analysis results.

**Request Body**:
```json
{
  "analysis_id": "ora_12345",
  "plot_types": ["dot_plot", "volcano_plot", "network_plot"],
  "parameters": {
    "interactive": true,
    "format": "html",
    "theme": "light",
    "max_pathways": 50
  }
}
```

**Response**:
```json
{
  "visualization_id": "viz_12345",
  "status": "completed",
  "plots": [
    {
      "plot_type": "dot_plot",
      "url": "/api/v1/visualize/viz_12345/dot_plot.html",
      "format": "html"
    },
    {
      "plot_type": "volcano_plot", 
      "url": "/api/v1/visualize/viz_12345/volcano_plot.html",
      "format": "html"
    }
  ]
}
```

#### GET /visualize/{visualization_id}/{plot_type}

Download specific visualization.

**Response**: File download (HTML, PNG, SVG, PDF)

### Job Management

#### GET /jobs

List user's jobs.

**Query Parameters**:
- `status`: Filter by status (pending, running, completed, failed)
- `limit`: Number of jobs to return (default: 50)
- `offset`: Offset for pagination (default: 0)

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "job_12345",
      "type": "ora",
      "status": "completed",
      "created_at": "2024-01-01T12:00:00Z",
      "completed_at": "2024-01-01T12:05:00Z",
      "progress": 100
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

#### GET /jobs/{job_id}

Get job details.

**Response**:
```json
{
  "job_id": "job_12345",
  "type": "ora",
  "status": "completed",
  "created_at": "2024-01-01T12:00:00Z",
  "completed_at": "2024-01-01T12:05:00Z",
  "progress": 100,
  "parameters": {
    "genes": ["TP53", "BRCA1"],
    "species": "human",
    "databases": ["kegg", "reactome"]
  },
  "results": {
    "analysis_id": "ora_12345"
  }
}
```

#### DELETE /jobs/{job_id}

Cancel a job.

**Response**:
```json
{
  "message": "Job cancelled successfully"
}
```

### System Information

#### GET /info

Get system information.

**Response**:
```json
{
  "version": "2.0.0",
  "status": "healthy",
  "databases": {
    "kegg": {"status": "available", "version": "2024.01"},
    "reactome": {"status": "available", "version": "2024.01"},
    "go": {"status": "available", "version": "2024.01"}
  },
  "cache": {
    "enabled": true,
    "size_mb": 150.5,
    "hit_rate": 0.85
  }
}
```

#### GET /info/databases

Get database information.

**Response**:
```json
{
  "databases": [
    {
      "name": "kegg",
      "display_name": "KEGG",
      "status": "available",
      "version": "2024.01",
      "species": ["human", "mouse", "rat"],
      "pathway_count": 15000
    }
  ]
}
```

### Configuration

#### GET /config

Get user configuration.

**Response**:
```json
{
  "analysis": {
    "significance_threshold": 0.05,
    "correction_method": "fdr_bh",
    "min_pathway_size": 5,
    "max_pathway_size": 500
  },
  "databases": {
    "kegg": {"enabled": true},
    "reactome": {"enabled": true},
    "go": {"enabled": true}
  },
  "cache": {
    "enabled": true,
    "ttl_days": 90
  }
}
```

#### PUT /config

Update user configuration.

**Request Body**:
```json
{
  "analysis": {
    "significance_threshold": 0.01,
    "correction_method": "fdr_bh"
  },
  "databases": {
    "kegg": {"enabled": true},
    "reactome": {"enabled": false}
  }
}
```

**Response**:
```json
{
  "message": "Configuration updated successfully"
}
```

## Data Models

### Gene

```json
{
  "gene_id": "string",
  "gene_symbol": "string",
  "ensembl_id": "string",
  "entrez_id": "string",
  "uniprot_id": "string",
  "species": "string",
  "description": "string"
}
```

### Pathway

```json
{
  "pathway_id": "string",
  "pathway_name": "string",
  "database": "string",
  "species": "string",
  "genes": ["string"],
  "description": "string",
  "url": "string"
}
```

### Analysis Result

```json
{
  "pathway_id": "string",
  "pathway_name": "string",
  "database": "string",
  "p_value": "number",
  "adjusted_p_value": "number",
  "genes": ["string"],
  "gene_ratio": "string",
  "background_ratio": "string",
  "enrichment_score": "number",
  "normalized_enrichment_score": "number"
}
```

### Job

```json
{
  "job_id": "string",
  "type": "string",
  "status": "string",
  "created_at": "string",
  "completed_at": "string",
  "progress": "number",
  "parameters": "object",
  "results": "object",
  "error": "string"
}
```

## WebSocket Events

PathwayLens 2.0 provides real-time updates via WebSocket connections:

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Events

#### Job Progress

```json
{
  "event": "job_progress",
  "data": {
    "job_id": "job_12345",
    "progress": 75,
    "status": "running",
    "message": "Processing pathways..."
  }
}
```

#### Job Completion

```json
{
  "event": "job_completed",
  "data": {
    "job_id": "job_12345",
    "status": "completed",
    "results": {
      "analysis_id": "ora_12345"
    }
  }
}
```

#### Job Error

```json
{
  "event": "job_error",
  "data": {
    "job_id": "job_12345",
    "status": "failed",
    "error": "Database connection failed"
  }
}
```

## SDKs and Libraries

### Python SDK

```python
from pathwaylens import PathwayLensClient

client = PathwayLensClient(
    base_url="http://localhost:8000/api/v1",
    access_token="your_access_token"
)

# Normalize genes
result = client.normalize(
    genes=["TP53", "BRCA1"],
    species="human",
    target_type="ensembl"
)

# Perform ORA analysis
analysis = client.analyze_ora(
    genes=["TP53", "BRCA1"],
    species="human",
    databases=["kegg", "reactome"]
)
```

### JavaScript SDK

```javascript
import { PathwayLensClient } from 'pathwaylens-js';

const client = new PathwayLensClient({
  baseUrl: 'http://localhost:8000/api/v1',
  accessToken: 'your_access_token'
});

// Normalize genes
const result = await client.normalize({
  genes: ['TP53', 'BRCA1'],
  species: 'human',
  targetType: 'ensembl'
});

// Perform ORA analysis
const analysis = await client.analyzeOra({
  genes: ['TP53', 'BRCA1'],
  species: 'human',
  databases: ['kegg', 'reactome']
});
```

## Examples

### Complete Workflow

```python
from pathwaylens import PathwayLensClient

client = PathwayLensClient(
    base_url="http://localhost:8000/api/v1",
    access_token="your_access_token"
)

# 1. Normalize genes
normalized = client.normalize(
    genes=["TP53", "BRCA1", "MYC"],
    species="human",
    target_type="ensembl"
)

# 2. Perform ORA analysis
ora_result = client.analyze_ora(
    genes=normalized["normalized_data"],
    species="human",
    databases=["kegg", "reactome", "go"]
)

# 3. Generate visualizations
viz_result = client.visualize(
    analysis_id=ora_result["analysis_id"],
    plot_types=["dot_plot", "volcano_plot"]
)

# 4. Compare with another dataset
comparison = client.compare(
    comparison_type="pathway_concordance",
    datasets=[
        {"name": "condition1", "analysis_id": ora_result["analysis_id"]},
        {"name": "condition2", "analysis_id": "ora_12346"}
    ]
)
```

### Error Handling

```python
from pathwaylens import PathwayLensClient, PathwayLensError

client = PathwayLensClient(
    base_url="http://localhost:8000/api/v1",
    access_token="your_access_token"
)

try:
    result = client.analyze_ora(
        genes=["TP53", "BRCA1"],
        species="human",
        databases=["kegg", "reactome"]
    )
except PathwayLensError as e:
    print(f"Error: {e.message}")
    print(f"Details: {e.details}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Rate Limits and Best Practices

### Rate Limiting

- **Authenticated users**: 1000 requests per hour
- **Anonymous users**: 100 requests per hour
- **Burst limit**: 100 requests per minute

### Best Practices

1. **Use batch endpoints** for multiple operations
2. **Cache results** to avoid repeated requests
3. **Handle errors gracefully** with proper error handling
4. **Use WebSocket connections** for real-time updates
5. **Monitor rate limits** and implement backoff strategies
6. **Validate input data** before sending requests
7. **Use appropriate timeouts** for long-running operations

## Support

- **Documentation**: Check the `docs/` directory
- **API Status**: Visit `/api/v1/info` for system status
- **Community**: Join our discussion forum
- **Support**: Contact our support team
