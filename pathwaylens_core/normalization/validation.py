"""
Input validation for various data formats and types.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from loguru import logger
import re

from .schemas import SpeciesType, IDType, ValidationResult


class InputValidator:
    """Validates input data for various formats and types."""
    
    def __init__(self):
        self.logger = logger.bind(module="input_validator")
        
        # Common gene ID patterns
        self.id_patterns = {
            IDType.SYMBOL: r'^[A-Za-z][A-Za-z0-9]*$',
            IDType.ENSEMBL: r'^ENS[A-Z]*G\d{11}$',
            IDType.ENTREZ: r'^\d+$',
            IDType.UNIPROT: r'^[A-Z0-9]{6}$|^[A-Z0-9]{10}$',
            IDType.REFSEQ: r'^[NX][MR]_\d+$',
            IDType.MGI: r'^MGI:\d+$',
            IDType.FLYBASE: r'^FBgn\d{7}$',
            IDType.WORMBASE: r'^WBGene\d{8}$',
            IDType.SGD: r'^Y[A-Z]{2}\d{3}[CW]?$'
        }
        
        # Species-specific ID patterns
        self.species_id_patterns = {
            SpeciesType.HUMAN: {
                IDType.SYMBOL: r'^[A-Za-z][A-Za-z0-9]*$',
                IDType.ENSEMBL: r'^ENSG\d{11}$',
                IDType.ENTREZ: r'^\d+$'
            },
            SpeciesType.MOUSE: {
                IDType.SYMBOL: r'^[A-Za-z][A-Za-z0-9]*$',
                IDType.ENSEMBL: r'^ENSMUSG\d{11}$',
                IDType.ENTREZ: r'^\d+$'
            },
            SpeciesType.RAT: {
                IDType.SYMBOL: r'^[A-Za-z][A-Za-z0-9]*$',
                IDType.ENSEMBL: r'^ENSRNOG\d{11}$',
                IDType.ENTREZ: r'^\d+$'
            }
        }
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate an input file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            ValidationResult with validation information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"File not found: {file_path}"]
            )
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            return ValidationResult(
                is_valid=False,
                errors=["File is empty"]
            )
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return ValidationResult(
                is_valid=False,
                errors=["File too large (>100MB)"]
            )
        
        # Try to read the file
        try:
            df = self._read_file(file_path)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Could not read file: {e}"]
            )
        
        # Validate DataFrame
        return self.validate_dataframe(df, file_path.name)
    
    def validate_dataframe(self, df: pd.DataFrame, filename: str = "data") -> ValidationResult:
        """
        Validate a pandas DataFrame.
        
        Args:
            df: DataFrame to validate
            filename: Name of the file (for error messages)
            
        Returns:
            ValidationResult with validation information
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check if DataFrame is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                file_info={"filename": filename, "rows": 0, "columns": 0}
            )
        
        # Check for required columns
        if len(df.columns) == 0:
            errors.append("No columns found")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                file_info={"filename": filename, "rows": len(df), "columns": 0}
            )
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            errors.append(f"Duplicate columns found: {duplicate_cols}")
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isnull().all()].tolist()
        if empty_cols:
            warnings.append(f"Empty columns found: {empty_cols}")
            suggestions.append("Consider removing empty columns")
        
        # Check for rows with all NaN values
        all_nan_rows = df.isnull().all(axis=1).sum()
        if all_nan_rows > 0:
            warnings.append(f"{all_nan_rows} rows contain only NaN values")
            suggestions.append("Consider removing rows with all NaN values")
        
        # Try to detect species and ID type
        detected_species = self._detect_species(df)
        detected_id_type = self._detect_id_type(df)
        
        # Validate gene IDs if detected
        if detected_id_type:
            id_validation = self._validate_gene_ids(df, detected_id_type, detected_species)
            errors.extend(id_validation['errors'])
            warnings.extend(id_validation['warnings'])
            suggestions.extend(id_validation['suggestions'])
        
        # Check for potential issues
        self._check_potential_issues(df, errors, warnings, suggestions)
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            detected_species=detected_species,
            detected_id_type=detected_id_type,
            file_info={
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "dtypes": df.dtypes.to_dict()
            }
        )
    
    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read file and return DataFrame."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix == '.tsv':
            return pd.read_csv(file_path, sep='\t')
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif suffix == '.feather':
            return pd.read_feather(file_path)
        else:
            # Try CSV as default
            return pd.read_csv(file_path)
    
    def _detect_species(self, df: pd.DataFrame) -> Optional[SpeciesType]:
        """Detect species from DataFrame content."""
        # Look for species-specific patterns in the data
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100)
                
                # Check for human Ensembl IDs
                if sample_values.str.match(r'^ENSG\d{11}$').any():
                    return SpeciesType.HUMAN
                
                # Check for mouse Ensembl IDs
                if sample_values.str.match(r'^ENSMUSG\d{11}$').any():
                    return SpeciesType.MOUSE
                
                # Check for rat Ensembl IDs
                if sample_values.str.match(r'^ENSRNOG\d{11}$').any():
                    return SpeciesType.RAT
        
        return None
    
    def _detect_id_type(self, df: pd.DataFrame) -> Optional[IDType]:
        """Detect ID type from DataFrame content."""
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100)
                
                # Check for Ensembl IDs
                if sample_values.str.match(r'^ENS[A-Z]*G\d{11}$').any():
                    return IDType.ENSEMBL
                
                # Check for Entrez IDs
                if sample_values.str.match(r'^\d+$').any():
                    return IDType.ENTREZ
                
                # Check for gene symbols
                if sample_values.str.match(r'^[A-Za-z][A-Za-z0-9]*$').any():
                    return IDType.SYMBOL
        
        return None
    
    def _validate_gene_ids(self, df: pd.DataFrame, id_type: IDType, species: Optional[SpeciesType]) -> Dict[str, List[str]]:
        """Validate gene IDs in the DataFrame."""
        errors = []
        warnings = []
        suggestions = []
        
        # Find the column with gene IDs
        id_column = None
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100)
                if self._matches_id_type(sample_values, id_type, species):
                    id_column = col
                    break
        
        if not id_column:
            errors.append("Could not find column with gene IDs")
            return {"errors": errors, "warnings": warnings, "suggestions": suggestions}
        
        # Validate IDs
        id_values = df[id_column].dropna()
        
        # Check for duplicates
        duplicates = id_values[id_values.duplicated()].unique()
        if len(duplicates) > 0:
            warnings.append(f"Found {len(duplicates)} duplicate gene IDs")
            suggestions.append("Consider removing duplicate entries")
        
        # Check for invalid IDs
        pattern = self._get_id_pattern(id_type, species)
        if pattern:
            invalid_ids = id_values[~id_values.str.match(pattern)].unique()
            if len(invalid_ids) > 0:
                warnings.append(f"Found {len(invalid_ids)} potentially invalid gene IDs")
                suggestions.append("Review invalid gene IDs")
        
        # Check for missing values
        missing_count = df[id_column].isnull().sum()
        if missing_count > 0:
            warnings.append(f"Found {missing_count} missing gene IDs")
            suggestions.append("Consider removing rows with missing gene IDs")
        
        return {"errors": errors, "warnings": warnings, "suggestions": suggestions}
    
    def _matches_id_type(self, values: pd.Series, id_type: IDType, species: Optional[SpeciesType]) -> bool:
        """Check if values match the specified ID type."""
        pattern = self._get_id_pattern(id_type, species)
        if not pattern:
            return False
        
        # Check if at least 80% of values match the pattern
        matches = values.str.match(pattern).sum()
        return matches / len(values) >= 0.8
    
    def _get_id_pattern(self, id_type: IDType, species: Optional[SpeciesType]) -> Optional[str]:
        """Get regex pattern for ID type and species."""
        if species and species in self.species_id_patterns:
            return self.species_id_patterns[species].get(id_type)
        return self.id_patterns.get(id_type)
    
    def _check_potential_issues(self, df: pd.DataFrame, errors: List[str], warnings: List[str], suggestions: List[str]):
        """Check for potential issues in the DataFrame."""
        # Check for very large numbers (potential scientific notation issues)
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].abs().max() > 1e10:
                warnings.append(f"Column '{col}' contains very large numbers")
                suggestions.append("Check if scientific notation is properly formatted")
        
        # Check for mixed data types in object columns
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                # Check if column contains mixed types
                types = df[col].apply(type).value_counts()
                if len(types) > 1:
                    warnings.append(f"Column '{col}' contains mixed data types")
                    suggestions.append("Consider converting to consistent data type")
        
        # Check for potential encoding issues
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                # Check for non-ASCII characters
                non_ascii = df[col].str.contains(r'[^\x00-\x7F]', na=False).sum()
                if non_ascii > 0:
                    warnings.append(f"Column '{col}' contains non-ASCII characters")
                    suggestions.append("Check file encoding")
        
        # Check for potential delimiter issues
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].dtype == 'object':
                # Check for commas in values (potential CSV parsing issues)
                commas = df[col].str.contains(',', na=False).sum()
                if commas > 0:
                    warnings.append(f"Column '{col}' contains commas in values")
                    suggestions.append("Check if file delimiter is correct")
    
    def validate_analysis_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate analysis parameters.
        
        Args:
            parameters: Dictionary of analysis parameters
            
        Returns:
            ValidationResult with validation information
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Check required parameters
        required_params = ['species', 'databases']
        for param in required_params:
            if param not in parameters:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate species
        if 'species' in parameters:
            species = parameters['species']
            if not isinstance(species, SpeciesType):
                try:
                    SpeciesType(species)
                except ValueError:
                    errors.append(f"Invalid species: {species}")
        
        # Validate databases
        if 'databases' in parameters:
            databases = parameters['databases']
            if not isinstance(databases, list):
                errors.append("Databases must be a list")
            elif len(databases) == 0:
                errors.append("At least one database must be specified")
        
        # Validate thresholds
        if 'significance_threshold' in parameters:
            threshold = parameters['significance_threshold']
            if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                errors.append("Significance threshold must be between 0 and 1")
        
        if 'min_pathway_size' in parameters:
            min_size = parameters['min_pathway_size']
            if not isinstance(min_size, int) or min_size < 1:
                errors.append("Minimum pathway size must be a positive integer")
        
        if 'max_pathway_size' in parameters:
            max_size = parameters['max_pathway_size']
            if not isinstance(max_size, int) or max_size < 1:
                errors.append("Maximum pathway size must be a positive integer")
        
        # Check for conflicting parameters
        if 'min_pathway_size' in parameters and 'max_pathway_size' in parameters:
            if parameters['min_pathway_size'] > parameters['max_pathway_size']:
                errors.append("Minimum pathway size cannot be greater than maximum pathway size")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
