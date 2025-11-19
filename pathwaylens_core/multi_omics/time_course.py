"""
Time course analysis module for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from loguru import logger

from ..analysis.schemas import AnalysisResult, AnalysisParameters
from ..data import DatabaseManager


class TimeCourseAnalyzer:
    """Time course analysis engine for PathwayLens."""
    
    def __init__(self, database_manager: Optional[DatabaseManager] = None):
        """
        Initialize the time course analyzer.
        
        Args:
            database_manager: Database manager instance
        """
        self.logger = logger.bind(module="time_course_analyzer")
        self.database_manager = database_manager or DatabaseManager()
        
        # Time course analysis methods
        self.analysis_methods = {
            'differential_expression': 'Differential Expression',
            'trajectory_analysis': 'Trajectory Analysis',
            'clustering': 'Time Course Clustering',
            'correlation': 'Time Course Correlation'
        }
    
    async def analyze_time_course(
        self,
        time_course_data: pd.DataFrame,
        parameters: AnalysisParameters,
        time_points: List[float],
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze time course data.
        
        Args:
            time_course_data: Time course expression data
            parameters: Analysis parameters
            time_points: List of time points
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with time course analysis
        """
        self.logger.info(f"Starting time course analysis with {len(time_points)} time points")
        
        try:
            # Validate input data
            validation_result = await self._validate_time_course_data(time_course_data, time_points)
            if not validation_result['valid']:
                raise ValueError(f"Invalid time course data: {validation_result['errors']}")
            
            # Extract genes with significant time course patterns
            significant_genes = await self._identify_significant_genes(time_course_data, time_points)
            
            # Perform pathway analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            result = await analysis_engine.analyze(
                significant_genes, parameters, output_dir
            )
            
            # Add time course-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'time_course',
                'time_points': time_points,
                'significant_genes': len(significant_genes),
                'time_course_statistics': await self._calculate_time_course_statistics(time_course_data, time_points)
            })
            
            self.logger.info("Time course analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Time course analysis failed: {e}")
            raise
    
    async def _validate_time_course_data(
        self, 
        time_course_data: pd.DataFrame, 
        time_points: List[float]
    ) -> Dict[str, Any]:
        """Validate time course data."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if time_course_data.empty:
            validation_result['valid'] = False
            validation_result['errors'].append("Time course data is empty")
            return validation_result
        
        # Check for gene ID column
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        gene_col = None
        
        for col in gene_columns:
            if col in time_course_data.columns:
                gene_col = col
                break
        
        if gene_col is None:
            validation_result['valid'] = False
            validation_result['errors'].append("No gene ID column found")
            return validation_result
        
        # Check for time point columns
        time_columns = [col for col in time_course_data.columns if col != gene_col]
        if len(time_columns) != len(time_points):
            validation_result['warnings'].append(f"Number of time columns ({len(time_columns)}) doesn't match time points ({len(time_points)})")
        
        return validation_result
    
    async def _identify_significant_genes(
        self, 
        time_course_data: pd.DataFrame, 
        time_points: List[float]
    ) -> List[str]:
        """Identify genes with significant time course patterns."""
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        gene_col = None
        
        for col in gene_columns:
            if col in time_course_data.columns:
                gene_col = col
                break
        
        if gene_col is None:
            return []
        
        # Get expression data
        expression_data = time_course_data.drop(columns=[gene_col])
        
        # Calculate time course statistics
        significant_genes = []
        
        for idx, row in time_course_data.iterrows():
            gene_id = row[gene_col]
            expression_values = row.drop(gene_col).values
            
            # Check for significant time course pattern
            if await self._is_significant_time_course(expression_values, time_points):
                significant_genes.append(gene_id)
        
        return significant_genes

    async def calculate_slopes(
        self,
        time_course_data: pd.DataFrame,
        time_points: List[float]
    ) -> Dict[str, float]:
        """Calculate expression slopes for each gene over time."""
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        gene_col = None
        
        for col in gene_columns:
            if col in time_course_data.columns:
                gene_col = col
                break
        
        if gene_col is None:
            return {}
        
        slopes = {}
        for idx, row in time_course_data.iterrows():
            gene_id = row[gene_col]
            expression_values = row.drop(gene_col).values
            
            if len(expression_values) >= 2:
                # Calculate slope using linear regression
                slope = np.polyfit(time_points[:len(expression_values)], expression_values, 1)[0]
                slopes[gene_id] = slope
        
        return slopes

    async def build_dynamic_network(
        self,
        time_course_data: pd.DataFrame,
        time_points: List[float],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Build dynamic network from time course data."""
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        gene_col = None
        
        for col in gene_columns:
            if col in time_course_data.columns:
                gene_col = col
                break
        
        if gene_col is None:
            return {"nodes": [], "edges": []}
        
        # Calculate correlations between genes over time
        expression_data = time_course_data.drop(columns=[gene_col])
        correlation_matrix = expression_data.corr()
        
        # Build network
        nodes = []
        edges = []
        
        for idx, row in time_course_data.iterrows():
            gene_id = row[gene_col]
            nodes.append({
                "id": gene_id,
                "label": gene_id,
                "slope": await self._get_gene_slope(gene_id, time_course_data, time_points)
            })
        
        # Add edges based on correlation
        for i, gene1 in enumerate(time_course_data[gene_col]):
            for j, gene2 in enumerate(time_course_data[gene_col]):
                if i < j:  # Avoid duplicates
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) > threshold:
                        edges.append({
                            "source": gene1,
                            "target": gene2,
                            "weight": abs(corr),
                            "correlation": corr
                        })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "time_points": time_points,
            "threshold": threshold
        }

    async def _get_gene_slope(
        self,
        gene_id: str,
        time_course_data: pd.DataFrame,
        time_points: List[float]
    ) -> float:
        """Get slope for a specific gene."""
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        gene_col = None
        
        for col in gene_columns:
            if col in time_course_data.columns:
                gene_col = col
                break
        
        if gene_col is None:
            return 0.0
        
        gene_row = time_course_data[time_course_data[gene_col] == gene_id]
        if gene_row.empty:
            return 0.0
        
        expression_values = gene_row.iloc[0].drop(gene_col).values
        if len(expression_values) >= 2:
            return np.polyfit(time_points[:len(expression_values)], expression_values, 1)[0]
        
        return 0.0

    async def _is_significant_time_course(
        self, 
        expression_values: np.ndarray, 
        time_points: List[float]
    ) -> bool:
        """Check if a gene has a significant time course pattern."""
        # Simplified significance test
        # In practice, would use proper statistical tests
        
        # Check for significant variation
        if len(expression_values) < 2:
            return False
        
        # Calculate coefficient of variation
        cv = np.std(expression_values) / np.mean(expression_values) if np.mean(expression_values) != 0 else 0
        
        # Check for trend
        if len(expression_values) >= 3:
            # Simple linear trend test
            correlation = np.corrcoef(time_points[:len(expression_values)], expression_values)[0, 1]
            if abs(correlation) > 0.5:  # Significant correlation
                return True
        
        # Check for high variation
        if cv > 0.3:  # High coefficient of variation
            return True
        
        return False
    
    async def _calculate_time_course_statistics(
        self, 
        time_course_data: pd.DataFrame, 
        time_points: List[float]
    ) -> Dict[str, Any]:
        """Calculate time course statistics."""
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        gene_col = None
        
        for col in gene_columns:
            if col in time_course_data.columns:
                gene_col = col
                break
        
        if gene_col is None:
            return {}
        
        # Get expression data
        expression_data = time_course_data.drop(columns=[gene_col])
        
        stats = {
            'total_genes': len(time_course_data),
            'time_points': time_points,
            'expression_statistics': {},
            'time_course_patterns': {}
        }
        
        # Calculate expression statistics
        for col in expression_data.columns:
            col_data = expression_data[col].dropna()
            stats['expression_statistics'][col] = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            }
        
        # Calculate time course patterns
        stats['time_course_patterns'] = await self._identify_time_course_patterns(time_course_data, time_points)
        
        return stats
    
    async def _identify_time_course_patterns(
        self, 
        time_course_data: pd.DataFrame, 
        time_points: List[float]
    ) -> Dict[str, int]:
        """Identify time course patterns."""
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        gene_col = None
        
        for col in gene_columns:
            if col in time_course_data.columns:
                gene_col = col
                break
        
        if gene_col is None:
            return {}
        
        patterns = {
            'increasing': 0,
            'decreasing': 0,
            'peak': 0,
            'trough': 0,
            'oscillating': 0,
            'stable': 0
        }
        
        for idx, row in time_course_data.iterrows():
            expression_values = row.drop(gene_col).values
            
            if len(expression_values) < 2:
                continue
            
            # Identify pattern
            pattern = await self._classify_time_course_pattern(expression_values, time_points)
            patterns[pattern] += 1
        
        return patterns
    
    async def _classify_time_course_pattern(
        self, 
        expression_values: np.ndarray, 
        time_points: List[float]
    ) -> str:
        """Classify time course pattern."""
        if len(expression_values) < 2:
            return 'stable'
        
        # Calculate trend
        correlation = np.corrcoef(time_points[:len(expression_values)], expression_values)[0, 1]
        
        if correlation > 0.5:
            return 'increasing'
        elif correlation < -0.5:
            return 'decreasing'
        
        # Check for peak or trough
        if len(expression_values) >= 3:
            max_idx = np.argmax(expression_values)
            min_idx = np.argmin(expression_values)
            
            if max_idx not in [0, len(expression_values)-1]:
                return 'peak'
            elif min_idx not in [0, len(expression_values)-1]:
                return 'trough'
        
        # Check for oscillation
        if len(expression_values) >= 4:
            # Simple oscillation detection
            diffs = np.diff(expression_values)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            if sign_changes > 1:
                return 'oscillating'
        
        return 'stable'
    
    async def analyze_differential_time_course(
        self,
        time_course_data: pd.DataFrame,
        parameters: AnalysisParameters,
        time_points: List[float],
        group_column: str,
        output_dir: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze differential time course between groups.
        
        Args:
            time_course_data: Time course data with group information
            parameters: Analysis parameters
            time_points: List of time points
            group_column: Column containing group information
            output_dir: Output directory for results
            
        Returns:
            AnalysisResult with differential time course analysis
        """
        self.logger.info("Starting differential time course analysis")
        
        try:
            # Validate input data
            validation_result = await self._validate_differential_time_course_data(
                time_course_data, time_points, group_column
            )
            if not validation_result['valid']:
                raise ValueError(f"Invalid differential time course data: {validation_result['errors']}")
            
            # Identify differentially expressed genes
            diff_genes = await self._identify_differential_genes(
                time_course_data, time_points, group_column
            )
            
            # Perform pathway analysis
            from ..analysis.engine import AnalysisEngine
            analysis_engine = AnalysisEngine(self.database_manager)
            
            result = await analysis_engine.analyze(
                diff_genes, parameters, output_dir
            )
            
            # Add differential time course-specific metadata
            result.metadata = result.metadata or {}
            result.metadata.update({
                'data_type': 'differential_time_course',
                'time_points': time_points,
                'group_column': group_column,
                'differential_genes': len(diff_genes),
                'group_statistics': await self._calculate_group_statistics(time_course_data, group_column)
            })
            
            self.logger.info("Differential time course analysis completed")
            return result
            
        except Exception as e:
            self.logger.error(f"Differential time course analysis failed: {e}")
            raise
    
    async def _validate_differential_time_course_data(
        self, 
        time_course_data: pd.DataFrame, 
        time_points: List[float], 
        group_column: str
    ) -> Dict[str, Any]:
        """Validate differential time course data."""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if time_course_data.empty:
            validation_result['valid'] = False
            validation_result['errors'].append("Time course data is empty")
            return validation_result
        
        if group_column not in time_course_data.columns:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Group column {group_column} not found")
            return validation_result
        
        # Check for multiple groups
        groups = time_course_data[group_column].unique()
        if len(groups) < 2:
            validation_result['valid'] = False
            validation_result['errors'].append("At least 2 groups required for differential analysis")
        
        return validation_result
    
    async def _identify_differential_genes(
        self, 
        time_course_data: pd.DataFrame, 
        time_points: List[float], 
        group_column: str
    ) -> List[str]:
        """Identify differentially expressed genes between groups."""
        gene_columns = ['gene_id', 'gene', 'gene_symbol', 'symbol']
        gene_col = None
        
        for col in gene_columns:
            if col in time_course_data.columns:
                gene_col = col
                break
        
        if gene_col is None:
            return []
        
        # Get groups
        groups = time_course_data[group_column].unique()
        
        if len(groups) < 2:
            return []
        
        # Calculate differential expression for each gene
        diff_genes = []
        
        for idx, row in time_course_data.iterrows():
            gene_id = row[gene_col]
            
            # Get expression values for each group
            group_expressions = {}
            for group in groups:
                group_data = time_course_data[time_course_data[group_column] == group]
                if len(group_data) > 0:
                    group_expressions[group] = group_data.iloc[0].drop([gene_col, group_column]).values
            
            # Check for significant difference
            if await self._is_significantly_different(group_expressions):
                diff_genes.append(gene_id)
        
        return diff_genes
    
    async def _is_significantly_different(self, group_expressions: Dict[str, np.ndarray]) -> bool:
        """Check if groups are significantly different."""
        if len(group_expressions) < 2:
            return False
        
        # Simplified significance test
        # In practice, would use proper statistical tests like ANOVA
        
        group_means = [np.mean(expr) for expr in group_expressions.values()]
        
        # Check for significant difference in means
        if len(group_means) >= 2:
            max_mean = max(group_means)
            min_mean = min(group_means)
            
            # Significant if difference is large relative to variation
            if max_mean - min_mean > 0.5:  # Simplified threshold
                return True
        
        return False
    
    async def _calculate_group_statistics(
        self, 
        time_course_data: pd.DataFrame, 
        group_column: str
    ) -> Dict[str, Any]:
        """Calculate group statistics."""
        groups = time_course_data[group_column].unique()
        
        stats = {
            'total_groups': len(groups),
            'group_sizes': {},
            'group_names': list(groups)
        }
        
        for group in groups:
            group_data = time_course_data[time_course_data[group_column] == group]
            stats['group_sizes'][group] = len(group_data)
        
        return stats
