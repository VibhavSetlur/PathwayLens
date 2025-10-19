"""
Database utilities for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
import sqlite3
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import hashlib
from datetime import datetime
from loguru import logger


class DatabaseUtils:
    """Database utility functions for PathwayLens."""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database utilities.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.logger = logger.bind(module="database_utils")
        self.db_path = db_path or ":memory:"
        self.connection = None
    
    async def connect(self) -> sqlite3.Connection:
        """Connect to the database."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
        return self.connection
    
    async def disconnect(self):
        """Disconnect from the database."""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    async def create_tables(self):
        """Create database tables if they don't exist."""
        conn = await self.connect()
        
        # Create jobs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                parameters TEXT NOT NULL,
                input_files TEXT,
                output_files TEXT,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create job_results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_results (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                result_type TEXT NOT NULL,
                result_data TEXT NOT NULL,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
            )
        """)
        
        # Create analysis_results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_results (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                species TEXT NOT NULL,
                input_gene_count INTEGER,
                total_pathways INTEGER,
                significant_pathways INTEGER,
                database_results TEXT,
                consensus_results TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
            )
        """)
        
        # Create pathway_results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pathway_results (
                id TEXT PRIMARY KEY,
                analysis_result_id TEXT NOT NULL,
                pathway_id TEXT NOT NULL,
                pathway_name TEXT NOT NULL,
                database TEXT NOT NULL,
                p_value REAL,
                adjusted_p_value REAL,
                enrichment_score REAL,
                normalized_enrichment_score REAL,
                overlap_count INTEGER,
                pathway_count INTEGER,
                input_count INTEGER,
                overlapping_genes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (analysis_result_id) REFERENCES analysis_results (id) ON DELETE CASCADE
            )
        """)
        
        # Create comparison_results table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS comparison_results (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                comparison_type TEXT NOT NULL,
                input_analysis_ids TEXT NOT NULL,
                overlap_statistics TEXT,
                correlation_results TEXT,
                clustering_results TEXT,
                visualization_data TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs (id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_results_job_id ON analysis_results (job_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pathway_results_analysis_id ON pathway_results (analysis_result_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pathway_results_pathway_id ON pathway_results (pathway_id)")
        
        conn.commit()
    
    async def insert_job(
        self,
        job_id: str,
        job_type: str,
        parameters: Dict[str, Any],
        input_files: Optional[List[str]] = None
    ) -> bool:
        """Insert a new job into the database."""
        try:
            conn = await self.connect()
            
            conn.execute("""
                INSERT INTO jobs (id, job_type, parameters, input_files)
                VALUES (?, ?, ?, ?)
            """, (
                job_id,
                job_type,
                json.dumps(parameters),
                json.dumps(input_files) if input_files else None
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert job {job_id}: {e}")
            return False
    
    async def update_job_status(
        self,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
        output_files: Optional[List[str]] = None
    ) -> bool:
        """Update job status."""
        try:
            conn = await self.connect()
            
            # Determine which timestamp to update
            timestamp_field = None
            if status == 'running':
                timestamp_field = 'started_at'
            elif status in ['completed', 'failed']:
                timestamp_field = 'completed_at'
            
            # Build update query
            update_fields = ['status = ?', 'updated_at = CURRENT_TIMESTAMP']
            params = [status]
            
            if error_message:
                update_fields.append('error_message = ?')
                params.append(error_message)
            
            if output_files:
                update_fields.append('output_files = ?')
                params.append(json.dumps(output_files))
            
            if timestamp_field:
                update_fields.append(f'{timestamp_field} = CURRENT_TIMESTAMP')
            
            params.append(job_id)
            
            conn.execute(f"""
                UPDATE jobs 
                SET {', '.join(update_fields)}
                WHERE id = ?
            """, params)
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update job {job_id}: {e}")
            return False
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job information."""
        try:
            conn = await self.connect()
            
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get job {job_id}: {e}")
            return None
    
    async def get_jobs(
        self,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of jobs with optional filtering."""
        try:
            conn = await self.connect()
            
            query = "SELECT * FROM jobs WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if job_type:
                query += " AND job_type = ?"
                params.append(job_type)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            self.logger.error(f"Failed to get jobs: {e}")
            return []
    
    async def insert_analysis_result(
        self,
        analysis_result_id: str,
        job_id: str,
        analysis_type: str,
        species: str,
        input_gene_count: int,
        total_pathways: int,
        significant_pathways: int,
        database_results: Dict[str, Any],
        consensus_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Insert analysis result into database."""
        try:
            conn = await self.connect()
            
            conn.execute("""
                INSERT INTO analysis_results (
                    id, job_id, analysis_type, species, input_gene_count,
                    total_pathways, significant_pathways, database_results,
                    consensus_results, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_result_id,
                job_id,
                analysis_type,
                species,
                input_gene_count,
                total_pathways,
                significant_pathways,
                json.dumps(database_results),
                json.dumps(consensus_results) if consensus_results else None,
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert analysis result {analysis_result_id}: {e}")
            return False
    
    async def insert_pathway_results(
        self,
        analysis_result_id: str,
        pathway_results: List[Dict[str, Any]]
    ) -> bool:
        """Insert pathway results into database."""
        try:
            conn = await self.connect()
            
            for pathway in pathway_results:
                conn.execute("""
                    INSERT INTO pathway_results (
                        id, analysis_result_id, pathway_id, pathway_name, database,
                        p_value, adjusted_p_value, enrichment_score, normalized_enrichment_score,
                        overlap_count, pathway_count, input_count, overlapping_genes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"{analysis_result_id}_{pathway['pathway_id']}",
                    analysis_result_id,
                    pathway['pathway_id'],
                    pathway['pathway_name'],
                    pathway['database'],
                    pathway.get('p_value'),
                    pathway.get('adjusted_p_value'),
                    pathway.get('enrichment_score'),
                    pathway.get('normalized_enrichment_score'),
                    pathway.get('overlap_count'),
                    pathway.get('pathway_count'),
                    pathway.get('input_count'),
                    json.dumps(pathway.get('overlapping_genes', []))
                ))
            
            conn.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert pathway results: {e}")
            return False
    
    async def get_analysis_result(self, analysis_result_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis result from database."""
        try:
            conn = await self.connect()
            
            cursor = conn.execute("SELECT * FROM analysis_results WHERE id = ?", (analysis_result_id,))
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                # Parse JSON fields
                result['database_results'] = json.loads(result['database_results'])
                if result['consensus_results']:
                    result['consensus_results'] = json.loads(result['consensus_results'])
                if result['metadata']:
                    result['metadata'] = json.loads(result['metadata'])
                return result
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get analysis result {analysis_result_id}: {e}")
            return None
    
    async def get_pathway_results(
        self,
        analysis_result_id: str,
        database: Optional[str] = None,
        significance_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """Get pathway results for an analysis."""
        try:
            conn = await self.connect()
            
            query = "SELECT * FROM pathway_results WHERE analysis_result_id = ?"
            params = [analysis_result_id]
            
            if database:
                query += " AND database = ?"
                params.append(database)
            
            if significance_threshold:
                query += " AND adjusted_p_value <= ?"
                params.append(significance_threshold)
            
            query += " ORDER BY adjusted_p_value ASC"
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                result = dict(row)
                result['overlapping_genes'] = json.loads(result['overlapping_genes'])
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get pathway results: {e}")
            return []
    
    async def cleanup_old_jobs(self, days: int = 30) -> int:
        """Clean up old completed jobs."""
        try:
            conn = await self.connect()
            
            cursor = conn.execute("""
                DELETE FROM jobs 
                WHERE status IN ('completed', 'failed') 
                AND completed_at < datetime('now', '-{} days')
            """.format(days))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            self.logger.info(f"Cleaned up {deleted_count} old jobs")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old jobs: {e}")
            return 0
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            conn = await self.connect()
            
            stats = {}
            
            # Job statistics
            cursor = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
            job_stats = dict(cursor.fetchall())
            stats['jobs'] = job_stats
            
            # Analysis statistics
            cursor = conn.execute("SELECT analysis_type, COUNT(*) FROM analysis_results GROUP BY analysis_type")
            analysis_stats = dict(cursor.fetchall())
            stats['analyses'] = analysis_stats
            
            # Pathway statistics
            cursor = conn.execute("SELECT database, COUNT(*) FROM pathway_results GROUP BY database")
            pathway_stats = dict(cursor.fetchall())
            stats['pathways'] = pathway_stats
            
            # Total counts
            cursor = conn.execute("SELECT COUNT(*) FROM jobs")
            stats['total_jobs'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM analysis_results")
            stats['total_analyses'] = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM pathway_results")
            stats['total_pathways'] = cursor.fetchone()[0]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get database stats: {e}")
            return {}
    
    async def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            import shutil
            
            if self.db_path == ":memory:":
                self.logger.warning("Cannot backup in-memory database")
                return False
            
            shutil.copy2(self.db_path, backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup database: {e}")
            return False
    
    async def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup."""
        try:
            import shutil
            
            if self.db_path == ":memory:":
                self.logger.warning("Cannot restore to in-memory database")
                return False
            
            shutil.copy2(backup_path, self.db_path)
            self.logger.info(f"Database restored from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore database: {e}")
            return False
