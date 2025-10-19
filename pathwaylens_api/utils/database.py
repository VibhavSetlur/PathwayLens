"""
Database utilities for PathwayLens API.

This module provides database connection management and utilities.
"""

import asyncio
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text, Column, String, DateTime, Text, Integer, Float, Boolean, JSON, ForeignKey
import logging

from pathwaylens_core.utils.config import get_config

logger = logging.getLogger(__name__)

# SQLAlchemy base
Base = declarative_base()

# Global database engine and session factory
_engine: Optional[Any] = None
_session_factory: Optional[async_sessionmaker] = None


class DatabaseManager:
    """Database connection manager."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.session_factory = None
    
    async def initialize(self):
        """Initialize database connection."""
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    async def close(self):
        """Close database connection."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager."""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a raw SQL query."""
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})
            return [dict(row._mapping) for row in result.fetchall()]
    
    async def execute_scalar(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a scalar query."""
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})
            return result.scalar()
    
    async def health_check(self) -> bool:
        """Check database health."""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


async def initialize_database():
    """Initialize global database connection."""
    global _db_manager, _engine, _session_factory
    
    config = get_config()
    database_url = config.get("database", {}).get("url")
    
    if not database_url:
        raise ValueError("Database URL not configured")
    
    _db_manager = DatabaseManager(database_url)
    await _db_manager.initialize()
    
    _engine = _db_manager.engine
    _session_factory = _db_manager.session_factory


async def close_database():
    """Close global database connection."""
    global _db_manager
    
    if _db_manager:
        await _db_manager.close()
        _db_manager = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager."""
    if not _db_manager:
        raise RuntimeError("Database not initialized")
    return _db_manager


@asynccontextmanager
async def get_database_session():
    """Get database session context manager."""
    if not _db_manager:
        raise RuntimeError("Database not initialized")
    
    async with _db_manager.get_session() as session:
        yield session


async def get_database_health() -> bool:
    """Check database health."""
    if not _db_manager:
        return False
    
    return await _db_manager.health_check()


# Database models
class Job(Base):
    """Job model for database."""
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=True)
    project_id = Column(String, ForeignKey("projects.id"), nullable=True)
    job_type = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    parameters = Column(JSON, nullable=False)
    input_files = Column(JSON, nullable=True)
    output_files = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    progress = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class JobResult(Base):
    """Job result model for database."""
    __tablename__ = "job_results"
    
    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    result_type = Column(String, nullable=False)
    result_data = Column(JSON, nullable=False)
    file_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    """User model for database."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Project(Base):
    """Project model for database."""
    __tablename__ = "projects"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(String, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AnalysisResult(Base):
    """Analysis results linked to a job."""
    __tablename__ = "analysis_results"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    analysis_type = Column(String, nullable=False)
    species = Column(String, nullable=False)
    input_gene_count = Column(Integer, nullable=True)
    total_pathways = Column(Integer, nullable=True)
    significant_pathways = Column(Integer, nullable=True)
    database_results = Column(JSON, nullable=True)
    consensus_results = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PathwayResult(Base):
    """Per-pathway statistics associated with an analysis result."""
    __tablename__ = "pathway_results"

    id = Column(String, primary_key=True)
    analysis_result_id = Column(String, ForeignKey("analysis_results.id"), nullable=False)
    pathway_id = Column(String, nullable=False)
    pathway_name = Column(Text, nullable=False)
    database = Column(String, nullable=False)
    p_value = Column(Float, nullable=True)
    adjusted_p_value = Column(Float, nullable=True)
    enrichment_score = Column(Float, nullable=True)
    normalized_enrichment_score = Column(Float, nullable=True)
    overlap_count = Column(Integer, nullable=True)
    pathway_count = Column(Integer, nullable=True)
    input_count = Column(Integer, nullable=True)
    overlapping_genes = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ComparisonResult(Base):
    """Results of comparing multiple analyses/datasets."""
    __tablename__ = "comparison_results"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    comparison_type = Column(String, nullable=False)
    input_analysis_ids = Column(JSON, nullable=False)
    overlap_statistics = Column(JSON, nullable=True)
    correlation_results = Column(JSON, nullable=True)
    clustering_results = Column(JSON, nullable=True)
    visualization_data = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Import required SQLAlchemy components (kept at end to avoid circular imports above)
from sqlalchemy import Column, String, Text, JSON, DateTime, ForeignKey, Integer, Float
from datetime import datetime
