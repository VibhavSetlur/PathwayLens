"""
Validation utilities for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from loguru import logger


class ValidationUtils:
    """Validation utility functions for PathwayLens."""
    
    def __init__(self):
        """Initialize validation utilities."""
        self.logger = logger.bind(module="validation_utils")
        
        # Gene ID patterns
        self.gene_id_patterns = {
            'ensembl_gene': r'^ENS[A-Z]*G\d{11}$',
            'ensembl_transcript': r'^ENS[A-Z]*T\d{11}$',
            'ensembl_protein': r'^ENS[A-Z]*P\d{11}$',
            'entrez': r'^\d+$',
            'uniprot': r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$',
            'refseq': r'^[NX][MR]_\d+\.\d+$',
            'genbank': r'^[A-Z]{1,2}_\d+\.\d+$',
            'symbol': r'^[A-Za-z][A-Za-z0-9]*$',
            'alias': r'^[A-Za-z][A-Za-z0-9]*$'
        }
        
        # Species validation
        self.valid_species = {
            'human', 'mouse', 'rat', 'zebrafish', 'drosophila', 
            'c_elegans', 'yeast', 'arabidopsis'
        }
        
        # Database validation
        self.valid_databases = {
            'kegg', 'reactome', 'go', 'biocyc', 'pathway_commons',
            'msigdb', 'panther', 'wikipathways'
        }
    
    async def validate_gene_ids(
        self,
        gene_ids: List[str],
        expected_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate gene IDs.
        
        Args:
            gene_ids: List of gene IDs to validate
            expected_type: Expected gene ID type
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Validating {len(gene_ids)} gene IDs")
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {},
            'detected_type': None
        }
        
        if not gene_ids:
            validation_result['valid'] = False
            validation_result['errors'].append("No gene IDs provided")
            return validation_result
        
        # Remove duplicates and empty values
        unique_ids = list(set([str(id).strip() for id in gene_ids if str(id).strip()]))
        
        if len(unique_ids) != len(gene_ids):
            validation_result['warnings'].append(f"Removed {len(gene_ids) - len(unique_ids)} duplicate/empty IDs")
        
        # Detect gene ID type
        detected_type = await self._detect_gene_id_type(unique_ids)
        validation_result['detected_type'] = detected_type
        
        # Validate against expected type
        if expected_type and detected_type != expected_type:
            validation_result['warnings'].append(f"Expected {expected_type}, detected {detected_type}")
        
        # Validate individual IDs
        invalid_ids = []
        for gene_id in unique_ids:
            if not self._is_valid_gene_id(gene_id, detected_type):
                invalid_ids.append(gene_id)
        
        if invalid_ids:
            validation_result['warnings'].append(f"Found {len(invalid_ids)} invalid IDs")
            validation_result['invalid_ids'] = invalid_ids[:10]  # Show first 10
        
        # Calculate statistics
        validation_result['statistics'] = {
            'total_ids': len(gene_ids),
            'unique_ids': len(unique_ids),
            'valid_ids': len(unique_ids) - len(invalid_ids),
            'invalid_ids': len(invalid_ids),
            'validity_rate': (len(unique_ids) - len(invalid_ids)) / len(unique_ids) if unique_ids else 0
        }
        
        return validation_result
    
    async def _detect_gene_id_type(self, gene_ids: List[str]) -> str:
        """Detect the most likely gene ID type."""
        type_scores = {}
        
        for id_type, pattern in self.gene_id_patterns.items():
            matches = sum(1 for gene_id in gene_ids if re.match(pattern, gene_id))
            type_scores[id_type] = matches / len(gene_ids) if gene_ids else 0
        
        # Return the type with highest score
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def _is_valid_gene_id(self, gene_id: str, id_type: str) -> bool:
        """Check if a gene ID is valid for the given type."""
        pattern = self.gene_id_patterns.get(id_type)
        if pattern:
            return bool(re.match(pattern, gene_id))
        return False
    
    async def validate_species(self, species: str) -> Dict[str, Any]:
        """
        Validate species name.
        
        Args:
            species: Species name to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not species:
            validation_result['valid'] = False
            validation_result['errors'].append("Species not provided")
            return validation_result
        
        species_lower = species.lower().strip()
        
        if species_lower not in self.valid_species:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Unsupported species: {species}")
            validation_result['suggestions'] = list(self.valid_species)
        
        return validation_result
    
    async def validate_databases(self, databases: List[str]) -> Dict[str, Any]:
        """
        Validate database names.
        
        Args:
            databases: List of database names to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not databases:
            validation_result['valid'] = False
            validation_result['errors'].append("No databases provided")
            return validation_result
        
        invalid_databases = []
        for database in databases:
            if database.lower() not in self.valid_databases:
                invalid_databases.append(database)
        
        if invalid_databases:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Invalid databases: {invalid_databases}")
            validation_result['suggestions'] = list(self.valid_databases)
        
        return validation_result
    
    async def validate_analysis_parameters(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate analysis parameters.
        
        Args:
            parameters: Analysis parameters to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate required parameters
        required_params = ['analysis_type', 'databases', 'species']
        for param in required_params:
            if param not in parameters:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Missing required parameter: {param}")
        
        # Validate analysis type
        if 'analysis_type' in parameters:
            valid_types = ['ora', 'gsea', 'gsva', 'topology', 'multi_omics', 'consensus']
            if parameters['analysis_type'] not in valid_types:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Invalid analysis type: {parameters['analysis_type']}")
        
        # Validate databases
        if 'databases' in parameters:
            db_validation = await self.validate_databases(parameters['databases'])
            if not db_validation['valid']:
                validation_result['valid'] = False
                validation_result['errors'].extend(db_validation['errors'])
        
        # Validate species
        if 'species' in parameters:
            species_validation = await self.validate_species(parameters['species'])
            if not species_validation['valid']:
                validation_result['valid'] = False
                validation_result['errors'].extend(species_validation['errors'])
        
        # Validate numerical parameters
        numerical_params = {
            'significance_threshold': (0.0, 1.0),
            'min_pathway_size': (1, 10000),
            'max_pathway_size': (1, 10000),
            'gsea_permutations': (100, 10000),
            'gsea_min_size': (1, 1000),
            'gsea_max_size': (1, 1000)
        }
        
        for param, (min_val, max_val) in numerical_params.items():
            if param in parameters:
                value = parameters[param]
                if not isinstance(value, (int, float)):
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Parameter {param} must be a number")
                elif value < min_val or value > max_val:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Parameter {param} must be between {min_val} and {max_val}")
        
        # Validate correction method
        if 'correction_method' in parameters:
            valid_methods = [
                'bonferroni', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky',
                'holm', 'hochberg', 'hommel', 'sidak', 'sidak_ss', 'sidak_sd'
            ]
            if parameters['correction_method'] not in valid_methods:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Invalid correction method: {parameters['correction_method']}")
        
        # Validate consensus method
        if 'consensus_method' in parameters:
            valid_methods = [
                'stouffer', 'fisher', 'brown', 'kost', 'tippett', 'mudholkar_george'
            ]
            if parameters['consensus_method'] not in valid_methods:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Invalid consensus method: {parameters['consensus_method']}")
        
        return validation_result
    
    async def validate_input_data(
        self,
        data: Union[pd.DataFrame, List[str], str],
        data_type: str = 'gene_list'
    ) -> Dict[str, Any]:
        """
        Validate input data for analysis.
        
        Args:
            data: Input data to validate
            data_type: Type of data ('gene_list', 'expression_matrix', 'differential_expression')
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        if data_type == 'gene_list':
            if isinstance(data, list):
                gene_validation = await self.validate_gene_ids(data)
                validation_result.update(gene_validation)
            elif isinstance(data, pd.DataFrame):
                # Find gene column
                gene_column = self._find_gene_column(data)
                if gene_column:
                    gene_ids = data[gene_column].dropna().tolist()
                    gene_validation = await self.validate_gene_ids(gene_ids)
                    validation_result.update(gene_validation)
                else:
                    validation_result['valid'] = False
                    validation_result['errors'].append("No gene column found in DataFrame")
            else:
                validation_result['valid'] = False
                validation_result['errors'].append("Invalid data type for gene list")
        
        elif data_type == 'expression_matrix':
            if isinstance(data, pd.DataFrame):
                # Check for numeric columns
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) == 0:
                    validation_result['valid'] = False
                    validation_result['errors'].append("No numeric columns found in expression matrix")
                
                # Check for missing values
                missing_count = data.isnull().sum().sum()
                if missing_count > 0:
                    validation_result['warnings'].append(f"Found {missing_count} missing values")
                
                validation_result['statistics'] = {
                    'num_genes': len(data),
                    'num_samples': len(data.columns),
                    'numeric_columns': len(numeric_columns),
                    'missing_values': missing_count
                }
            else:
                validation_result['valid'] = False
                validation_result['errors'].append("Expression matrix must be a DataFrame")
        
        elif data_type == 'differential_expression':
            if isinstance(data, pd.DataFrame):
                # Check for required columns
                required_columns = ['gene_id', 'log2fc', 'p_value']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Missing required columns: {missing_columns}")
                
                # Validate gene IDs
                if 'gene_id' in data.columns:
                    gene_ids = data['gene_id'].dropna().tolist()
                    gene_validation = await self.validate_gene_ids(gene_ids)
                    validation_result.update(gene_validation)
                
                validation_result['statistics'] = {
                    'num_genes': len(data),
                    'columns': data.columns.tolist()
                }
            else:
                validation_result['valid'] = False
                validation_result['errors'].append("Differential expression data must be a DataFrame")
        
        return validation_result
    
    def _find_gene_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the most likely gene ID column in a DataFrame."""
        # Common column names for gene IDs
        gene_column_names = [
            'gene_id', 'gene', 'gene_symbol', 'symbol', 'gene_name',
            'ensembl_id', 'ensembl_gene_id', 'entrez_id', 'entrez',
            'uniprot_id', 'uniprot', 'refseq_id', 'refseq'
        ]
        
        # Check for exact matches first
        for col in df.columns:
            if col.lower() in [name.lower() for name in gene_column_names]:
                return col
        
        # Check for partial matches
        for col in df.columns:
            col_lower = col.lower()
            if any(name in col_lower for name in ['gene', 'id', 'symbol']):
                return col
        
        # If no obvious match, return the first column
        return df.columns[0] if len(df.columns) > 0 else None
    
    async def validate_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        Validate file path.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not file_path:
            validation_result['valid'] = False
            validation_result['errors'].append("File path not provided")
            return validation_result
        
        from pathlib import Path
        
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            validation_result['valid'] = False
            validation_result['errors'].append("File does not exist")
            return validation_result
        
        # Check if it's a file (not directory)
        if not path.is_file():
            validation_result['valid'] = False
            validation_result['errors'].append("Path is not a file")
            return validation_result
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:  # 100 MB limit
            validation_result['warnings'].append(f"Large file size: {file_size_mb:.2f} MB")
        
        # Check file extension
        valid_extensions = ['.csv', '.tsv', '.txt', '.xlsx', '.xls', '.json']
        if path.suffix.lower() not in valid_extensions:
            validation_result['warnings'].append(f"Unusual file extension: {path.suffix}")
        
        return validation_result
    
    async def validate_output_directory(self, output_dir: str) -> Dict[str, Any]:
        """
        Validate output directory.
        
        Args:
            output_dir: Output directory path to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        if not output_dir:
            validation_result['valid'] = False
            validation_result['errors'].append("Output directory not provided")
            return validation_result
        
        from pathlib import Path
        
        path = Path(output_dir)
        
        # Check if directory exists
        if not path.exists():
            # Try to create directory
            try:
                path.mkdir(parents=True, exist_ok=True)
                validation_result['warnings'].append("Created output directory")
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Cannot create output directory: {str(e)}")
                return validation_result
        
        # Check if it's a directory
        if not path.is_dir():
            validation_result['valid'] = False
            validation_result['errors'].append("Path is not a directory")
            return validation_result
        
        # Check write permissions
        try:
            test_file = path / 'test_write_permission.tmp'
            test_file.write_text('test')
            test_file.unlink()
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"No write permission: {str(e)}")
        
        return validation_result
