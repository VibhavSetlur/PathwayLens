"""
Main normalization engine for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid
from loguru import logger

from .format_detector import FormatDetector
from .id_converter import IDConverter
from .species_mapper import SpeciesMapper
from .validation import InputValidator
from .schemas import (
    NormalizedTable, NormalizationResult, SpeciesType, IDType, 
    AmbiguityPolicy, ValidationResult
)


class Normalizer:
    """Main normalization engine for gene identifier conversion and data processing."""
    
    def __init__(self, rate_limit: float = 1.0):
        """
        Initialize the normalizer.
        
        Args:
            rate_limit: Rate limit for API calls (requests per second)
        """
        self.rate_limit = rate_limit
        self.logger = logger.bind(module="normalizer")
        
        # Initialize components
        self.format_detector = FormatDetector()
        self.id_converter = IDConverter(rate_limit)
        self.species_mapper = SpeciesMapper(rate_limit)
        self.validator = InputValidator()
    
    async def normalize_file(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        target_id_type: IDType = IDType.SYMBOL,
        target_species: Optional[SpeciesType] = None,
        ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.EXPAND,
        validate_input: bool = True
    ) -> NormalizationResult:
        """
        Normalize a file containing gene identifiers.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file (optional)
            target_id_type: Target identifier type
            target_species: Target species (optional)
            ambiguity_policy: How to handle ambiguous mappings
            validate_input: Whether to validate input before processing
            
        Returns:
            NormalizationResult with normalization information
        """
        job_id = str(uuid.uuid4())
        start_time = datetime.now().isoformat()
        
        self.logger.info(f"Starting normalization job {job_id} for file {input_file}")
        
        try:
            # Step 1: Validate input file
            if validate_input:
                validation_result = self.validator.validate_file(input_file)
                if not validation_result.is_valid:
                    return NormalizationResult(
                        job_id=job_id,
                        input_file=input_file,
                        species=target_species or SpeciesType.HUMAN,
                        input_id_type=IDType.SYMBOL,  # Will be detected
                        output_id_type=target_id_type,
                        total_input=0,
                        total_mapped=0,
                        total_unmapped=0,
                        mapping_rate=0.0,
                        errors=validation_result.errors,
                        created_at=start_time
                    )
            
            # Step 2: Detect file format
            format_result = self.format_detector.detect_format(input_file)
            if format_result.confidence < 0.5:
                return NormalizationResult(
                    job_id=job_id,
                    input_file=input_file,
                    species=target_species or SpeciesType.HUMAN,
                    input_id_type=IDType.SYMBOL,
                    output_id_type=target_id_type,
                    total_input=0,
                    total_mapped=0,
                    total_unmapped=0,
                    mapping_rate=0.0,
                    errors=[f"Could not detect file format: {format_result.errors}"],
                    created_at=start_time
                )
            
            # Step 3: Read and parse file
            df = await self._read_file(input_file, format_result)
            if df.empty:
                return NormalizationResult(
                    job_id=job_id,
                    input_file=input_file,
                    species=target_species or SpeciesType.HUMAN,
                    input_id_type=IDType.SYMBOL,
                    output_id_type=target_id_type,
                    total_input=0,
                    total_mapped=0,
                    total_unmapped=0,
                    mapping_rate=0.0,
                    errors=["File is empty or could not be parsed"],
                    created_at=start_time
                )
            
            # Step 4: Detect species and ID type
            detected_species = target_species or self._detect_species(df)
            detected_id_type = self._detect_id_type(df)
            
            if not detected_id_type:
                return NormalizationResult(
                    job_id=job_id,
                    input_file=input_file,
                    species=detected_species,
                    input_id_type=IDType.SYMBOL,
                    output_id_type=target_id_type,
                    total_input=len(df),
                    total_mapped=0,
                    total_unmapped=len(df),
                    mapping_rate=0.0,
                    errors=["Could not detect identifier type"],
                    created_at=start_time
                )
            
            # Step 5: Extract gene identifiers
            gene_ids = self._extract_gene_ids(df, detected_id_type)
            if not gene_ids:
                return NormalizationResult(
                    job_id=job_id,
                    input_file=input_file,
                    species=detected_species,
                    input_id_type=detected_id_type,
                    output_id_type=target_id_type,
                    total_input=len(df),
                    total_mapped=0,
                    total_unmapped=len(df),
                    mapping_rate=0.0,
                    errors=["No gene identifiers found"],
                    created_at=start_time
                )
            
            # Step 6: Convert identifiers
            conversion_results = await self._convert_identifiers(
                gene_ids, detected_id_type, target_id_type, 
                detected_species, target_species, ambiguity_policy
            )
            
            # Step 7: Create normalized table
            normalized_table = await self._create_normalized_table(
                df, conversion_results, detected_id_type, target_id_type
            )
            
            # Step 8: Save output file
            if output_file:
                await self._save_output(normalized_table, output_file)
            
            # Step 9: Calculate statistics
            total_input = len(gene_ids)
            total_mapped = len([r for r in conversion_results if r.output_id is not None])
            total_unmapped = total_input - total_mapped
            mapping_rate = total_mapped / total_input if total_input > 0 else 0.0
            
            ambiguous_mappings = len([r for r in conversion_results if r.is_ambiguous])
            duplicate_mappings = len([r for r in conversion_results if r.output_id is not None]) - len(set(r.output_id for r in conversion_results if r.output_id is not None))
            
            end_time = datetime.now().isoformat()
            
            self.logger.info(f"Normalization job {job_id} completed successfully")
            
            return NormalizationResult(
                job_id=job_id,
                input_file=input_file,
                output_file=output_file,
                species=detected_species,
                input_id_type=detected_id_type,
                output_id_type=target_id_type,
                total_input=total_input,
                total_mapped=total_mapped,
                total_unmapped=total_unmapped,
                mapping_rate=mapping_rate,
                ambiguous_mappings=ambiguous_mappings,
                duplicate_mappings=duplicate_mappings,
                warnings=format_result.errors or [],
                created_at=start_time,
                completed_at=end_time
            )
            
        except Exception as e:
            self.logger.error(f"Normalization job {job_id} failed: {e}")
            return NormalizationResult(
                job_id=job_id,
                input_file=input_file,
                species=target_species or SpeciesType.HUMAN,
                input_id_type=IDType.SYMBOL,
                output_id_type=target_id_type,
                total_input=0,
                total_mapped=0,
                total_unmapped=0,
                mapping_rate=0.0,
                errors=[f"Normalization failed: {str(e)}"],
                created_at=start_time
            )
    
    async def _read_file(self, file_path: str, format_result) -> pd.DataFrame:
        """Read file based on detected format."""
        file_path = Path(file_path)
        
        if format_result.format_type == 'csv':
            return pd.read_csv(file_path, delimiter=format_result.delimiter)
        elif format_result.format_type == 'tsv':
            return pd.read_csv(file_path, delimiter='\t')
        elif format_result.format_type == 'excel':
            return pd.read_excel(file_path)
        elif format_result.format_type == 'json':
            return pd.read_json(file_path)
        elif format_result.format_type == 'parquet':
            return pd.read_parquet(file_path)
        elif format_result.format_type == 'feather':
            return pd.read_feather(file_path)
        else:
            # Default to CSV
            return pd.read_csv(file_path)
    
    def _detect_species(self, df: pd.DataFrame) -> SpeciesType:
        """Detect species from DataFrame content."""
        # Look for species-specific patterns
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
        
        # Default to human
        return SpeciesType.HUMAN
    
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
    
    def _extract_gene_ids(self, df: pd.DataFrame, id_type: IDType) -> List[str]:
        """Extract gene identifiers from DataFrame."""
        gene_ids = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100)
                if self._matches_id_type(sample_values, id_type):
                    # This column contains the gene IDs
                    gene_ids = df[col].dropna().unique().tolist()
                    break
        
        return gene_ids
    
    def _matches_id_type(self, values: pd.Series, id_type: IDType) -> bool:
        """Check if values match the specified ID type."""
        patterns = {
            IDType.SYMBOL: r'^[A-Za-z][A-Za-z0-9]*$',
            IDType.ENSEMBL: r'^ENS[A-Z]*G\d{11}$',
            IDType.ENTREZ: r'^\d+$',
            IDType.UNIPROT: r'^[A-Z0-9]{6}$|^[A-Z0-9]{10}$',
            IDType.REFSEQ: r'^[NX][MR]_\d+$'
        }
        
        pattern = patterns.get(id_type)
        if not pattern:
            return False
        
        # Check if at least 80% of values match the pattern
        matches = values.str.match(pattern).sum()
        return matches / len(values) >= 0.8
    
    async def _convert_identifiers(
        self,
        gene_ids: List[str],
        input_id_type: IDType,
        output_id_type: IDType,
        source_species: SpeciesType,
        target_species: Optional[SpeciesType],
        ambiguity_policy: AmbiguityPolicy
    ) -> List[Any]:
        """Convert gene identifiers."""
        if target_species and target_species != source_species:
            # Cross-species mapping needed
            async with self.species_mapper as mapper:
                ortholog_results = await mapper.map_orthologs(
                    gene_ids, source_species, target_species
                )
                
                # Convert ortholog results to conversion results
                conversion_results = []
                for result in ortholog_results:
                    conversion_results.append(type('ConversionResult', (), {
                        'input_id': result.source_id,
                        'output_id': result.target_id,
                        'confidence': result.confidence,
                        'source': result.method,
                        'is_ambiguous': result.is_ambiguous,
                        'alternative_mappings': result.alternative_mappings
                    })())
                
                return conversion_results
        else:
            # Same species, direct conversion
            async with self.id_converter as converter:
                return await converter.convert_identifiers(
                    gene_ids, input_id_type, output_id_type, 
                    source_species, ambiguity_policy
                )
    
    async def _create_normalized_table(
        self,
        original_df: pd.DataFrame,
        conversion_results: List[Any],
        input_id_type: IDType,
        output_id_type: IDType
    ) -> NormalizedTable:
        """Create normalized table from conversion results."""
        # Create mapping dictionary
        id_mapping = {}
        for result in conversion_results:
            if result.output_id is not None:
                id_mapping[result.input_id] = result.output_id
        
        # Create new DataFrame with converted IDs
        normalized_df = original_df.copy()
        
        # Find the column with gene IDs and convert it
        for col in normalized_df.columns:
            if normalized_df[col].dtype == 'object':
                sample_values = normalized_df[col].dropna().head(100)
                if self._matches_id_type(sample_values, input_id_type):
                    # Convert this column
                    normalized_df[col] = normalized_df[col].map(id_mapping).fillna(normalized_df[col])
                    break
        
        # Create metadata
        metadata = {
            'input_id_type': input_id_type.value,
            'output_id_type': output_id_type.value,
            'conversion_mapping': id_mapping,
            'total_conversions': len(id_mapping),
            'conversion_rate': len(id_mapping) / len(conversion_results) if conversion_results else 0.0
        }
        
        return NormalizedTable(
            data=normalized_df,
            species=SpeciesType.HUMAN,  # Will be set properly
            id_type=output_id_type,
            metadata=metadata
        )
    
    async def _save_output(self, normalized_table: NormalizedTable, output_file: str):
        """Save normalized table to output file."""
        output_path = Path(output_file)
        suffix = output_path.suffix.lower()
        
        if suffix == '.csv':
            normalized_table.data.to_csv(output_file, index=False)
        elif suffix == '.tsv':
            normalized_table.data.to_csv(output_file, sep='\t', index=False)
        elif suffix in ['.xlsx', '.xls']:
            normalized_table.data.to_excel(output_file, index=False)
        elif suffix == '.json':
            normalized_table.data.to_json(output_file, orient='records', indent=2)
        elif suffix == '.parquet':
            normalized_table.data.to_parquet(output_file, index=False)
        elif suffix == '.feather':
            normalized_table.data.to_feather(output_file)
        else:
            # Default to CSV
            normalized_table.data.to_csv(output_file, index=False)
    
    def normalize_file_sync(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        target_id_type: IDType = IDType.SYMBOL,
        target_species: Optional[SpeciesType] = None,
        ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.EXPAND,
        validate_input: bool = True
    ) -> NormalizationResult:
        """
        Synchronous version of normalize_file.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file (optional)
            target_id_type: Target identifier type
            target_species: Target species (optional)
            ambiguity_policy: How to handle ambiguous mappings
            validate_input: Whether to validate input before processing
            
        Returns:
            NormalizationResult with normalization information
        """
        return asyncio.run(self.normalize_file(
            input_file, output_file, target_id_type, target_species, 
            ambiguity_policy, validate_input
        ))

    async def normalize_list(
        self,
        gene_list: List[str],
        input_type: Union[str, IDType] = IDType.SYMBOL,
        output_type: Union[str, IDType] = IDType.ENSEMBL,
        species: SpeciesType = SpeciesType.HUMAN,
        ambiguity_policy: AmbiguityPolicy = AmbiguityPolicy.EXPAND
    ) -> NormalizationResult:
        """
        Normalize a list of gene identifiers.
        
        Args:
            gene_list: List of gene identifiers
            input_type: Input identifier type
            output_type: Output identifier type
            species: Species
            ambiguity_policy: How to handle ambiguous mappings
            
        Returns:
            NormalizationResult with normalization information
        """
        job_id = str(uuid.uuid4())
        start_time = datetime.now().isoformat()
        
        # Convert string types to enums if needed
        if isinstance(input_type, str):
            try:
                input_type = IDType(input_type)
            except ValueError:
                # Try to map common names
                if input_type.lower() == "symbol":
                    input_type = IDType.SYMBOL
                elif input_type.lower() == "ensembl":
                    input_type = IDType.ENSEMBL
                elif input_type.lower() == "entrez":
                    input_type = IDType.ENTREZ
                else:
                    # Default or error
                    pass
                    
        if isinstance(output_type, str):
            try:
                output_type = IDType(output_type)
            except ValueError:
                # Try to map common names
                if output_type.lower() == "symbol":
                    output_type = IDType.SYMBOL
                elif output_type.lower() == "ensembl_gene_id":
                    output_type = IDType.ENSEMBL
                elif output_type.lower() == "entrez":
                    output_type = IDType.ENTREZ
                else:
                    # Default or error
                    pass

        try:
            # Convert identifiers
            conversion_results = await self._convert_identifiers(
                gene_list, input_type, output_type, 
                species, species, ambiguity_policy
            )
            
            # Calculate statistics
            total_input = len(gene_list)
            total_mapped = len([r for r in conversion_results if r.output_id is not None])
            total_unmapped = total_input - total_mapped
            mapping_rate = total_mapped / total_input if total_input > 0 else 0.0
            
            ambiguous_mappings = len([r for r in conversion_results if r.is_ambiguous])
            duplicate_mappings = len([r for r in conversion_results if r.output_id is not None]) - len(set(r.output_id for r in conversion_results if r.output_id is not None))
            
            # Create converted genes list (simple list of mapped IDs)
            converted_genes = [r.output_id for r in conversion_results if r.output_id is not None]
            
            # Create result object (extending NormalizationResult to include converted_genes for compatibility)
            result = NormalizationResult(
                job_id=job_id,
                input_file="in_memory_list",
                species=species,
                input_id_type=input_type,
                output_id_type=output_type,
                total_input=total_input,
                total_mapped=total_mapped,
                total_unmapped=total_unmapped,
                mapping_rate=mapping_rate,
                ambiguous_mappings=ambiguous_mappings,
                duplicate_mappings=duplicate_mappings,
                created_at=start_time,
                completed_at=datetime.now().isoformat()
            )
            
            
            return result
            
        except Exception as e:
            self.logger.error(f"List normalization failed: {e}")
            raise
