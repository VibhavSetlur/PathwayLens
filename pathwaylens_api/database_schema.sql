-- PathwayLens 2.0 Database Schema
-- PostgreSQL database schema for PathwayLens API

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Projects table
CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Jobs table
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    job_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    parameters JSONB NOT NULL,
    input_files JSONB,
    output_files JSONB,
    error_message TEXT,
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Job results table
CREATE TABLE job_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    result_type VARCHAR(50) NOT NULL,
    result_data JSONB NOT NULL,
    file_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Analysis results table
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    species VARCHAR(50) NOT NULL,
    input_gene_count INTEGER,
    total_pathways INTEGER,
    significant_pathways INTEGER,
    database_results JSONB,
    consensus_results JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Pathway results table
CREATE TABLE pathway_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analysis_result_id UUID REFERENCES analysis_results(id) ON DELETE CASCADE,
    pathway_id VARCHAR(255) NOT NULL,
    pathway_name TEXT NOT NULL,
    database VARCHAR(50) NOT NULL,
    p_value DECIMAL(10, 8),
    adjusted_p_value DECIMAL(10, 8),
    enrichment_score DECIMAL(10, 6),
    normalized_enrichment_score DECIMAL(10, 6),
    overlap_count INTEGER,
    pathway_count INTEGER,
    input_count INTEGER,
    overlapping_genes JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Comparison results table
CREATE TABLE comparison_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    comparison_type VARCHAR(50) NOT NULL,
    input_analysis_ids JSONB NOT NULL,
    overlap_statistics JSONB,
    correlation_results JSONB,
    clustering_results JSONB,
    visualization_data JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Project jobs relationship
CREATE TABLE project_jobs (
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (project_id, job_id)
);

-- Indexes for performance
CREATE INDEX idx_jobs_user_id ON jobs(user_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_job_type ON jobs(job_type);
CREATE INDEX idx_jobs_created_at ON jobs(created_at);
CREATE INDEX idx_jobs_project_id ON jobs(project_id);

CREATE INDEX idx_job_results_job_id ON job_results(job_id);
CREATE INDEX idx_job_results_result_type ON job_results(result_type);

CREATE INDEX idx_analysis_results_job_id ON analysis_results(job_id);
CREATE INDEX idx_analysis_results_analysis_type ON analysis_results(analysis_type);
CREATE INDEX idx_analysis_results_species ON analysis_results(species);

CREATE INDEX idx_pathway_results_analysis_result_id ON pathway_results(analysis_result_id);
CREATE INDEX idx_pathway_results_pathway_id ON pathway_results(pathway_id);
CREATE INDEX idx_pathway_results_database ON pathway_results(database);
CREATE INDEX idx_pathway_results_p_value ON pathway_results(p_value);

CREATE INDEX idx_comparison_results_job_id ON comparison_results(job_id);
CREATE INDEX idx_comparison_results_comparison_type ON comparison_results(comparison_type);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_projects_owner_id ON projects(owner_id);

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries
CREATE VIEW job_summary AS
SELECT 
    j.id,
    j.job_type,
    j.status,
    j.progress,
    j.created_at,
    j.completed_at,
    u.email as user_email,
    p.name as project_name,
    CASE 
        WHEN j.completed_at IS NOT NULL THEN 
            EXTRACT(EPOCH FROM (j.completed_at - j.created_at))
        ELSE NULL
    END as duration_seconds
FROM jobs j
LEFT JOIN users u ON j.user_id = u.id
LEFT JOIN projects p ON j.project_id = p.id;

CREATE VIEW analysis_summary AS
SELECT 
    ar.id,
    ar.analysis_type,
    ar.species,
    ar.input_gene_count,
    ar.total_pathways,
    ar.significant_pathways,
    j.created_at,
    u.email as user_email
FROM analysis_results ar
JOIN jobs j ON ar.job_id = j.id
LEFT JOIN users u ON j.user_id = u.id;

-- Sample data for development
INSERT INTO users (id, email, name, password_hash, role) VALUES
    (uuid_generate_v4(), 'admin@pathwaylens.com', 'Admin User', 'hashed_password', 'admin'),
    (uuid_generate_v4(), 'user@pathwaylens.com', 'Test User', 'hashed_password', 'user');

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pathwaylens_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pathwaylens_user;
