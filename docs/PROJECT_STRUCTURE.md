# PathwayLens 2.0 Project Structure

## Overview

PathwayLens 2.0 follows a modular, scalable architecture designed for maintainability and extensibility. This document describes the project structure, organization, and design principles.

## Directory Structure

```
PathwayLens/
├── Archive/                    # Archived PathwayLens 1.0 files
├── docs/                       # Documentation
├── pathwaylens_core/           # Core backend modules
├── pathwaylens_cli/            # Standalone CLI application
├── pathwaylens_api/            # FastAPI web application
├── frontend/                   # Next.js web interface
├── infra/                      # Infrastructure and deployment
├── tests/                      # Test suite
├── config/                     # Configuration files
├── pyproject.toml             # Python project configuration
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup script
├── setup.sh                   # Setup script
├── README.md                  # Project overview
├── LICENSE                    # MIT license
└── PATHWAYLENS_2.0_UPGRADE_BLUEPRINT.md  # Project blueprint
```

## Core Modules

### pathwaylens_core/

The core backend modules containing the main business logic and data processing capabilities.

```
pathwaylens_core/
├── __init__.py
├── normalization/              # Data normalization and conversion
│   ├── __init__.py
│   ├── schemas.py             # Pydantic data models
│   ├── format_detector.py     # Input format detection
│   ├── id_converter.py        # Gene ID conversion
│   ├── species_mapper.py      # Cross-species mapping
│   ├── validation.py          # Input validation
│   └── normalizer.py          # Main normalization engine
├── analysis/                  # Pathway analysis engines
│   ├── __init__.py
│   ├── schemas.py             # Analysis result models
│   └── engine.py              # Main analysis engine
├── data/                      # Database adapters and caching
│   ├── __init__.py
│   ├── adapters/              # Database adapters
│   │   ├── __init__.py
│   │   ├── base.py            # Base adapter class
│   │   ├── kegg_adapter.py    # KEGG database adapter
│   │   ├── reactome_adapter.py # Reactome adapter
│   │   ├── go_adapter.py      # Gene Ontology adapter
│   │   ├── biocyc_adapter.py  # BioCyc adapter
│   │   ├── pathway_commons_adapter.py # Pathway Commons adapter
│   │   ├── msigdb_adapter.py  # MSigDB adapter
│   │   ├── panther_adapter.py # PANTHER adapter
│   │   └── wikipathways_adapter.py # WikiPathways adapter
│   ├── cache/                 # Caching system
│   │   ├── __init__.py
│   │   ├── cache_manager.py   # Cache management
│   │   └── version_manager.py # Cache versioning
│   ├── database_manager.py    # Database orchestration
│   ├── mapping/               # Data mapping utilities
│   ├── utils/                 # Data utilities
│   └── utils/                 # General utilities
├── comparison/                # Multi-dataset comparison
│   ├── __init__.py
│   └── schemas.py             # Comparison result models
├── visualization/             # Visualization and reporting
│   ├── __init__.py
│   └── schemas.py             # Visualization models
├── multi_omics/               # Multi-omics analysis
│   ├── __init__.py
│   └── schemas.py             # Multi-omics models
├── plugins/                   # Plugin system
│   ├── __init__.py
│   └── base.py                # Plugin base class
└── utils/                     # Core utilities
    ├── __init__.py
    ├── config.py              # Configuration management
    ├── exceptions.py          # Custom exceptions
    └── logger.py              # Logging system
```

### pathwaylens_cli/

Standalone command-line interface built with Typer and Rich.

```
pathwaylens_cli/
├── __init__.py
├── main.py                    # Main CLI entry point
├── commands/                  # CLI commands
│   ├── __init__.py
│   ├── normalize.py           # Normalize command
│   ├── analyze.py             # Analyze command
│   ├── compare.py             # Compare command
│   ├── visualize.py           # Visualize command
│   ├── config.py              # Config command
│   └── info.py                # Info command
├── utils/                     # CLI utilities
│   ├── __init__.py
│   ├── formatters.py          # Output formatting
│   └── validators.py          # Input validation
├── templates/                 # CLI templates
│   └── config.yml             # Default configuration
└── config/                    # CLI configuration
    └── settings.py            # CLI settings
```

### pathwaylens_api/

FastAPI web application providing RESTful API endpoints.

```
pathwaylens_api/
├── __init__.py
├── main.py                    # FastAPI application
├── routes/                    # API routes
│   ├── __init__.py
│   ├── auth.py                # Authentication routes
│   ├── normalize.py           # Normalization endpoints
│   ├── analyze.py             # Analysis endpoints
│   ├── compare.py             # Comparison endpoints
│   ├── visualize.py           # Visualization endpoints
│   ├── jobs.py                # Job management endpoints
│   └── config.py              # Configuration endpoints
├── schemas/                   # API schemas
│   ├── __init__.py
│   ├── auth.py                # Authentication schemas
│   ├── normalize.py           # Normalization schemas
│   ├── analyze.py             # Analysis schemas
│   ├── compare.py             # Comparison schemas
│   ├── visualize.py           # Visualization schemas
│   └── jobs.py                # Job schemas
├── middleware/                # API middleware
│   ├── __init__.py
│   ├── auth.py                # Authentication middleware
│   ├── rate_limit.py          # Rate limiting
│   └── cors.py                # CORS handling
└── utils/                     # API utilities
    ├── __init__.py
    ├── dependencies.py        # FastAPI dependencies
    └── exceptions.py          # API exceptions
```

### frontend/

Next.js web interface with TypeScript and Tailwind CSS.

```
frontend/
├── package.json               # Node.js dependencies
├── next.config.js             # Next.js configuration
├── tailwind.config.js         # Tailwind CSS configuration
├── tsconfig.json              # TypeScript configuration
├── src/                       # Source code
│   ├── app/                   # Next.js app directory
│   │   ├── layout.tsx         # Root layout
│   │   ├── page.tsx           # Home page
│   │   ├── analyze/           # Analysis pages
│   │   ├── compare/           # Comparison pages
│   │   ├── visualize/         # Visualization pages
│   │   └── api/               # API routes
│   ├── components/            # Reusable components
│   │   ├── ui/                # UI components
│   │   ├── forms/             # Form components
│   │   ├── charts/            # Chart components
│   │   └── layout/            # Layout components
│   ├── hooks/                 # Custom React hooks
│   │   ├── useAnalysis.ts     # Analysis hooks
│   │   ├── useVisualization.ts # Visualization hooks
│   │   └── useAuth.ts         # Authentication hooks
│   ├── services/              # API services
│   │   ├── api.ts             # API client
│   │   ├── auth.ts            # Authentication service
│   │   └── analysis.ts        # Analysis service
│   ├── stores/                # State management
│   │   ├── analysis.ts        # Analysis state
│   │   ├── visualization.ts   # Visualization state
│   │   └── auth.ts            # Authentication state
│   ├── types/                 # TypeScript types
│   │   ├── analysis.ts        # Analysis types
│   │   ├── visualization.ts   # Visualization types
│   │   └── api.ts             # API types
│   └── utils/                 # Utility functions
│       ├── validation.ts      # Input validation
│       ├── formatting.ts      # Data formatting
│       └── constants.ts       # Constants
├── design-system/             # Design system
│   ├── components/            # Design system components
│   ├── tokens/                # Design tokens
│   └── styles/                # Global styles
└── public/                    # Static assets
    ├── icons/                 # Icons
    ├── images/                # Images
    └── fonts/                 # Fonts
```

### infra/

Infrastructure and deployment configuration.

```
infra/
├── docker/                    # Docker configuration
│   ├── Dockerfile.backend     # Backend Dockerfile
│   ├── Dockerfile.frontend    # Frontend Dockerfile
│   ├── docker-compose.yml     # Development setup
│   └── docker-compose.prod.yml # Production setup
├── kubernetes/                # Kubernetes configuration
│   ├── namespace.yaml         # Namespace
│   ├── backend/               # Backend deployment
│   ├── frontend/              # Frontend deployment
│   ├── database/              # Database deployment
│   └── ingress/               # Ingress configuration
├── monitoring/                # Monitoring setup
│   ├── prometheus/            # Prometheus configuration
│   ├── grafana/               # Grafana dashboards
│   └── alerts/                # Alert rules
├── ci-cd/                     # CI/CD pipelines
│   ├── github-actions/        # GitHub Actions workflows
│   ├── tests/                 # Test configurations
│   └── deployment/            # Deployment scripts
└── config/                    # Infrastructure configuration
    ├── environments/          # Environment configs
    └── secrets/               # Secret management
```

### tests/

Comprehensive test suite for all components.

```
tests/
├── __init__.py
├── conftest.py                # Pytest configuration
├── unit/                      # Unit tests
│   ├── test_normalization.py  # Normalization tests
│   ├── test_analysis.py       # Analysis tests
│   ├── test_data.py           # Data adapter tests
│   ├── test_comparison.py     # Comparison tests
│   └── test_visualization.py  # Visualization tests
├── integration/               # Integration tests
│   ├── test_api.py            # API integration tests
│   ├── test_cli.py            # CLI integration tests
│   └── test_workflows.py      # End-to-end workflows
├── e2e/                       # End-to-end tests
│   ├── test_web_interface.py  # Web interface tests
│   └── test_cli_workflows.py  # CLI workflow tests
└── fixtures/                  # Test fixtures
    ├── genes/                 # Gene test data
    ├── pathways/              # Pathway test data
    ├── analysis/              # Analysis test data
    └── configs/               # Test configurations
```

### docs/

Comprehensive documentation.

```
docs/
├── INSTALLATION.md            # Installation guide
├── QUICKSTART.md              # Quick start guide
├── USER_GUIDE.md              # User guide
├── CLI_REFERENCE.md           # CLI reference
├── API_REFERENCE.md           # API reference
├── CONTRIBUTING.md            # Contributing guide
├── CHANGELOG.md               # Release notes
├── PROJECT_STRUCTURE.md       # This file
└── examples/                  # Usage examples
    ├── basic_analysis/        # Basic analysis examples
    ├── multi_omics/           # Multi-omics examples
    └── advanced_features/     # Advanced feature examples
```

## Design Principles

### 1. Modular Architecture

- **Separation of Concerns**: Each module has a specific responsibility
- **Loose Coupling**: Modules interact through well-defined interfaces
- **High Cohesion**: Related functionality is grouped together
- **Extensibility**: New modules can be added without affecting existing ones

### 2. Scalability

- **Horizontal Scaling**: Support for multiple instances
- **Vertical Scaling**: Efficient resource utilization
- **Caching**: Intelligent caching for performance
- **Async Processing**: Non-blocking operations

### 3. Maintainability

- **Clear Structure**: Intuitive directory organization
- **Documentation**: Comprehensive documentation
- **Testing**: Extensive test coverage
- **Code Quality**: Consistent coding standards

### 4. Performance

- **Efficient Algorithms**: Optimized data processing
- **Resource Management**: Proper memory and CPU usage
- **Database Optimization**: Efficient database queries
- **Caching Strategy**: Strategic data caching

### 5. Security

- **Input Validation**: Comprehensive input sanitization
- **Authentication**: Secure user authentication
- **Authorization**: Role-based access control
- **Data Protection**: Secure data handling

## Module Dependencies

### Core Dependencies

```
pathwaylens_core
├── normalization
│   ├── data.adapters
│   ├── data.cache
│   └── utils
├── analysis
│   ├── data.adapters
│   ├── data.cache
│   └── utils
├── data
│   ├── adapters
│   ├── cache
│   └── utils
├── comparison
│   ├── analysis
│   └── utils
├── visualization
│   ├── analysis
│   └── utils
└── multi_omics
    ├── analysis
    ├── comparison
    └── utils
```

### CLI Dependencies

```
pathwaylens_cli
├── pathwaylens_core
├── typer
├── rich
└── click
```

### API Dependencies

```
pathwaylens_api
├── pathwaylens_core
├── fastapi
├── uvicorn
├── pydantic
└── sqlalchemy
```

### Frontend Dependencies

```
frontend
├── next.js
├── react
├── typescript
├── tailwindcss
├── plotly.js
├── d3.js
└── zustand
```

## Configuration Management

### Environment Configuration

- **Development**: Local development settings
- **Testing**: Test environment settings
- **Staging**: Pre-production settings
- **Production**: Production settings

### Configuration Files

- **pyproject.toml**: Python project configuration
- **package.json**: Node.js project configuration
- **docker-compose.yml**: Docker configuration
- **kubernetes/**: Kubernetes configuration
- **config/**: Application configuration

## Data Flow

### 1. Input Processing

```
User Input → Validation → Format Detection → Normalization → Analysis
```

### 2. Analysis Pipeline

```
Normalized Data → Database Query → Pathway Analysis → Result Processing → Output
```

### 3. Visualization Pipeline

```
Analysis Results → Data Processing → Visualization Generation → Interactive Output
```

## Error Handling

### Error Types

- **Validation Errors**: Input validation failures
- **Database Errors**: Database connection/query failures
- **Analysis Errors**: Analysis processing failures
- **System Errors**: System-level failures

### Error Handling Strategy

- **Graceful Degradation**: Continue operation when possible
- **User-Friendly Messages**: Clear error messages
- **Logging**: Comprehensive error logging
- **Recovery**: Automatic retry and recovery

## Testing Strategy

### Test Types

- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Performance and load testing

### Test Coverage

- **Code Coverage**: Minimum 80% code coverage
- **Branch Coverage**: All code branches tested
- **Integration Coverage**: All integration points tested
- **User Scenario Coverage**: All user scenarios tested

## Deployment Strategy

### Deployment Environments

- **Development**: Local development
- **Testing**: Automated testing
- **Staging**: Pre-production testing
- **Production**: Live production system

### Deployment Methods

- **Docker**: Containerized deployment
- **Kubernetes**: Orchestrated deployment
- **Cloud**: Cloud platform deployment
- **On-Premise**: Local server deployment

## Monitoring and Observability

### Monitoring Stack

- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization
- **ELK Stack**: Log aggregation
- **Jaeger**: Distributed tracing

### Key Metrics

- **Performance**: Response times, throughput
- **Reliability**: Error rates, availability
- **Usage**: User activity, feature usage
- **Resources**: CPU, memory, disk usage

## Security Considerations

### Security Measures

- **Authentication**: JWT-based authentication
- **Authorization**: Role-based access control
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: API rate limiting
- **HTTPS**: Secure communication
- **Secrets Management**: Secure secret handling

### Security Testing

- **Static Analysis**: Code security analysis
- **Dynamic Testing**: Runtime security testing
- **Penetration Testing**: Security vulnerability testing
- **Dependency Scanning**: Third-party dependency security

## Future Considerations

### Scalability Improvements

- **Microservices**: Service decomposition
- **Event-Driven Architecture**: Event-based communication
- **Caching**: Advanced caching strategies
- **Load Balancing**: Traffic distribution

### Feature Enhancements

- **Machine Learning**: ML-based analysis
- **Real-time Processing**: Stream processing
- **Advanced Visualizations**: 3D and VR visualizations
- **Mobile Support**: Mobile application

### Technology Updates

- **Framework Updates**: Regular framework updates
- **Dependency Updates**: Security and feature updates
- **Performance Optimizations**: Continuous performance improvements
- **New Technologies**: Adoption of new technologies
