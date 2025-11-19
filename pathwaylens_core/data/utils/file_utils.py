"""
File utilities for PathwayLens.
"""

import asyncio
import pandas as pd
import numpy as np
import json
import csv
import gzip
import zipfile
from typing import Dict, List, Optional, Any, Union, Tuple, Iterator
from pathlib import Path
import hashlib
import mimetypes
from datetime import datetime
from loguru import logger


class FileUtils:
    """File utility functions for PathwayLens."""
    
    def __init__(self):
        """Initialize file utilities."""
        self.logger = logger.bind(module="file_utils")
        
        # Supported file formats
        self.supported_formats = {
            'csv': ['.csv'],
            'tsv': ['.tsv', '.txt'],
            'excel': ['.xlsx', '.xls'],
            'json': ['.json'],
            'hdf5': ['.h5', '.hdf5'],
            'parquet': ['.parquet'],
            'pickle': ['.pkl', '.pickle'],
            'compressed': ['.gz', '.zip']
        }
        
        # Gene ID patterns for detection
        self.gene_id_patterns = {
            'ensembl_gene': r'^ENS[A-Z]*G\d{11}$',
            'ensembl_transcript': r'^ENS[A-Z]*T\d{11}$',
            'entrez': r'^\d+$',
            'symbol': r'^[A-Za-z][A-Za-z0-9]*$',
            'uniprot': r'^[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$'
        }
    
    async def detect_file_format(self, file_path: str) -> str:
        """
        Detect file format from file extension and content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected file format
        """
        file_path = Path(file_path)
        
        # Check extension first
        extension = file_path.suffix.lower()
        
        for format_name, extensions in self.supported_formats.items():
            if extension in extensions:
                return format_name
        
        # Try to detect from content
        try:
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                
                # Check for common file signatures
                if header.startswith(b'PK'):
                    return 'zip'
                elif header.startswith(b'\x1f\x8b'):
                    return 'gzip'
                elif header.startswith(b'{') or header.startswith(b'['):
                    return 'json'
                elif b'\t' in header:
                    return 'tsv'
                elif b',' in header:
                    return 'csv'
                    
        except Exception as e:
            self.logger.warning(f"Could not detect file format for {file_path}: {e}")
        
        return 'unknown'
    
    async def read_file(
        self,
        file_path: str,
        format: Optional[str] = None,
        streaming: bool = False,
        chunksize: Optional[int] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]], 'Iterator[pd.DataFrame]']:
        """
        Read file and return appropriate data structure.
        
        Args:
            file_path: Path to the file
            format: File format (auto-detected if None)
            streaming: If True, return iterator for chunked reading
            chunksize: Size of chunks when streaming (default: 10000)
            **kwargs: Additional arguments for pandas readers
            
        Returns:
            DataFrame, dictionary, list, or iterator depending on file type and streaming flag
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect format if not specified
        if format is None:
            format = await self.detect_file_format(str(file_path))
        
        self.logger.info(f"Reading {format} file: {file_path}")
        
        try:
            if format == 'csv':
                # Improved CSV reading with better error handling and memory efficiency
                # Use chunking for large files
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                use_chunks = file_size_mb > 100  # Use chunks for files > 100MB
                
                csv_kwargs = {
                    'low_memory': True if use_chunks else False,  # Use low_memory for large files
                    'on_bad_lines': 'skip',
                    'encoding': kwargs.pop('encoding', 'utf-8'),
                    **kwargs
                }
                
                # For streaming mode, return iterator
                if streaming:
                    chunk_size = chunksize or kwargs.pop('chunksize', 10000)
                    return pd.read_csv(file_path, chunksize=chunk_size, **csv_kwargs)
                
                # For large files, read in chunks and concatenate (memory-efficient)
                if use_chunks and 'chunksize' not in kwargs:
                    chunks = []
                    chunk_size = kwargs.pop('chunksize', 10000)
                    for chunk in pd.read_csv(file_path, chunksize=chunk_size, **csv_kwargs):
                        chunks.append(chunk)
                    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
                
                try:
                    return pd.read_csv(file_path, **csv_kwargs)
                except UnicodeDecodeError:
                    # Try with different encodings
                    for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                        try:
                            csv_kwargs['encoding'] = encoding
                            return pd.read_csv(file_path, **csv_kwargs)
                        except (UnicodeDecodeError, Exception):
                            continue
                    raise ValueError(f"Could not read CSV file with supported encodings: {file_path}")
            elif format == 'tsv':
                # Improved TSV reading with better delimiter detection
                tsv_kwargs = {
                    'sep': '\t',
                    'low_memory': False,
                    'on_bad_lines': 'skip',
                    'encoding': kwargs.pop('encoding', 'utf-8'),
                    **kwargs
                }
                try:
                    return pd.read_csv(file_path, **tsv_kwargs)
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    # Try to auto-detect delimiter
                    with open(file_path, 'r', encoding=tsv_kwargs['encoding']) as f:
                        sample = f.read(1024)
                    delimiters = ['\t', '|', ';', ',']
                    for delim in delimiters:
                        if delim in sample:
                            tsv_kwargs['sep'] = delim
                            try:
                                return pd.read_csv(file_path, **tsv_kwargs)
                            except Exception:
                                continue
                    raise
                except UnicodeDecodeError:
                    # Try with different encodings
                    for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                        try:
                            tsv_kwargs['encoding'] = encoding
                            return pd.read_csv(file_path, **tsv_kwargs)
                        except (UnicodeDecodeError, Exception):
                            continue
                    raise ValueError(f"Could not read TSV file with supported encodings: {file_path}")
            elif format == 'excel':
                # Improved Excel reading with sheet name support and better error handling
                excel_kwargs = {
                    'engine': 'openpyxl' if str(file_path).endswith('.xlsx') else None,
                    **kwargs
                }
                try:
                    return pd.read_excel(file_path, **excel_kwargs)
                except ImportError:
                    # Fallback if openpyxl not available
                    excel_kwargs.pop('engine', None)
                    return pd.read_excel(file_path, **excel_kwargs)
                except Exception as e:
                    # Try reading first sheet explicitly
                    try:
                        excel_kwargs['sheet_name'] = 0
                        return pd.read_excel(file_path, **excel_kwargs)
                    except Exception:
                        raise ValueError(f"Could not read Excel file: {file_path}. Error: {e}")
            elif format == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif format == 'hdf5':
                return pd.read_hdf(file_path, **kwargs)
            elif format == 'parquet':
                return pd.read_parquet(file_path, **kwargs)
            elif format == 'pickle':
                return pd.read_pickle(file_path, **kwargs)
            elif format == 'gzip':
                # Try to detect inner format
                with gzip.open(file_path, 'rt') as f:
                    sample = f.read(1024)
                    f.seek(0)
                    
                    if '\t' in sample:
                        return pd.read_csv(f, sep='\t', **kwargs)
                    else:
                        return pd.read_csv(f, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            raise
    
    async def write_file(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        file_path: str,
        format: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Write data to file.
        
        Args:
            data: Data to write
            file_path: Output file path
            format: File format (auto-detected if None)
            **kwargs: Additional arguments for pandas writers
            
        Returns:
            True if successful
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Detect format if not specified
        if format is None:
            format = await self.detect_file_format(str(file_path))
        
        self.logger.info(f"Writing {format} file: {file_path}")
        
        try:
            if format == 'csv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, index=False, **kwargs)
                else:
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        if isinstance(data, list) and data:
                            writer.writerow(data[0].keys())
                            for row in data:
                                writer.writerow(row.values())
            elif format == 'tsv':
                if isinstance(data, pd.DataFrame):
                    data.to_csv(file_path, sep='\t', index=False, **kwargs)
                else:
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f, delimiter='\t')
                        if isinstance(data, list) and data:
                            writer.writerow(data[0].keys())
                            for row in data:
                                writer.writerow(row.values())
            elif format == 'excel':
                if isinstance(data, pd.DataFrame):
                    data.to_excel(file_path, index=False, **kwargs)
                else:
                    df = pd.DataFrame(data)
                    df.to_excel(file_path, index=False, **kwargs)
            elif format == 'json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, **kwargs)
            elif format == 'hdf5':
                if isinstance(data, pd.DataFrame):
                    data.to_hdf(file_path, key='data', **kwargs)
                else:
                    df = pd.DataFrame(data)
                    df.to_hdf(file_path, key='data', **kwargs)
            elif format == 'parquet':
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(file_path, **kwargs)
                else:
                    df = pd.DataFrame(data)
                    df.to_parquet(file_path, **kwargs)
            elif format == 'pickle':
                if isinstance(data, pd.DataFrame):
                    data.to_pickle(file_path, **kwargs)
                else:
                    import pickle
                    with open(file_path, 'wb') as f:
                        pickle.dump(data, f, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {format}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write file {file_path}: {e}")
            return False
    
    async def detect_gene_id_type(self, file_path: str, column: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect gene ID type in a file.
        
        Args:
            file_path: Path to the file
            column: Column name to analyze (auto-detected if None)
            
        Returns:
            Dictionary with detection results
        """
        import re
        
        # Read file
        data = await self.read_file(file_path)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("File must contain tabular data for gene ID detection")
        
        # Find gene column if not specified
        if column is None:
            column = self._find_gene_column(data)
        
        if column is None:
            return {
                'detected_type': 'unknown',
                'confidence': 0.0,
                'sample_ids': [],
                'total_ids': 0
            }
        
        # Get sample of gene IDs
        gene_ids = data[column].dropna().astype(str).head(1000).tolist()
        
        # Test against patterns
        type_matches = {}
        for id_type, pattern in self.gene_id_patterns.items():
            matches = sum(1 for gene_id in gene_ids if re.match(pattern, gene_id))
            type_matches[id_type] = matches / len(gene_ids) if gene_ids else 0
        
        # Find best match
        best_type = max(type_matches.items(), key=lambda x: x[1])
        
        return {
            'detected_type': best_type[0],
            'confidence': best_type[1],
            'sample_ids': gene_ids[:10],
            'total_ids': len(gene_ids),
            'all_matches': type_matches
        }
    
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
    
    async def validate_file(
        self,
        file_path: str,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        max_size_mb: float = 100.0
    ) -> Dict[str, Any]:
        """
        Validate file for PathwayLens processing.
        
        Args:
            file_path: Path to the file
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            max_size_mb: Maximum file size in MB
            
        Returns:
            Dictionary with validation results
        """
        file_path = Path(file_path)
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {}
        }
        
        try:
            # Check file existence
            if not file_path.exists():
                validation_result['valid'] = False
                validation_result['errors'].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            validation_result['file_info']['size_mb'] = file_size_mb
            
            if file_size_mb > max_size_mb:
                validation_result['valid'] = False
                validation_result['errors'].append(f"File size ({file_size_mb:.2f} MB) exceeds maximum ({max_size_mb} MB)")
            
            # Try to read file
            try:
                data = await self.read_file(str(file_path))
                validation_result['file_info']['format'] = await self.detect_file_format(str(file_path))
            except Exception as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Cannot read file: {str(e)}")
                return validation_result
            
            # Check if it's tabular data
            if not isinstance(data, pd.DataFrame):
                validation_result['valid'] = False
                validation_result['errors'].append("File does not contain tabular data")
                return validation_result
            
            # Check number of rows
            num_rows = len(data)
            validation_result['file_info']['num_rows'] = num_rows
            
            if num_rows < min_rows:
                validation_result['valid'] = False
                validation_result['errors'].append(f"File has {num_rows} rows, minimum required is {min_rows}")
            
            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Missing required columns: {missing_columns}")
            
            # Check for empty values
            empty_columns = data.columns[data.isnull().all()].tolist()
            if empty_columns:
                validation_result['warnings'].append(f"Columns with all empty values: {empty_columns}")
            
            # Check for duplicate rows
            duplicate_rows = data.duplicated().sum()
            if duplicate_rows > 0:
                validation_result['warnings'].append(f"Found {duplicate_rows} duplicate rows")
            
            validation_result['file_info']['num_columns'] = len(data.columns)
            validation_result['file_info']['columns'] = data.columns.tolist()
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation failed: {str(e)}")
        
        return validation_result
    
    async def create_file_hash(self, file_path: str, algorithm: str = 'md5') -> str:
        """
        Create hash of file content.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
            
        Returns:
            File hash
        """
        file_path = Path(file_path)
        
        if algorithm == 'md5':
            hasher = hashlib.md5()
        elif algorithm == 'sha1':
            hasher = hashlib.sha1()
        elif algorithm == 'sha256':
            hasher = hashlib.sha256()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    async def compress_file(self, file_path: str, compression: str = 'gzip') -> str:
        """
        Compress a file.
        
        Args:
            file_path: Path to the file
            compression: Compression type ('gzip', 'zip')
            
        Returns:
            Path to compressed file
        """
        file_path = Path(file_path)
        
        if compression == 'gzip':
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        elif compression == 'zip':
            compressed_path = file_path.with_suffix('.zip')
            with zipfile.ZipFile(compressed_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_path, file_path.name)
        else:
            raise ValueError(f"Unsupported compression type: {compression}")
        
        return str(compressed_path)
    
    async def decompress_file(self, file_path: str) -> str:
        """
        Decompress a file.
        
        Args:
            file_path: Path to the compressed file
            
        Returns:
            Path to decompressed file
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.gz':
            decompressed_path = file_path.with_suffix('')
            with gzip.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        elif file_path.suffix == '.zip':
            decompressed_path = file_path.with_suffix('')
            with zipfile.ZipFile(file_path, 'r') as zipf:
                zipf.extractall(decompressed_path.parent)
        else:
            raise ValueError(f"Unsupported compressed file format: {file_path.suffix}")
        
        return str(decompressed_path)
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive file information.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': 'File not found'}
        
        stat = file_path.stat()
        
        info = {
            'name': file_path.name,
            'path': str(file_path),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'format': await self.detect_file_format(str(file_path)),
            'mime_type': mimetypes.guess_type(str(file_path))[0]
        }
        
        # Try to get additional info for tabular files
        try:
            data = await self.read_file(str(file_path))
            if isinstance(data, pd.DataFrame):
                info.update({
                    'num_rows': len(data),
                    'num_columns': len(data.columns),
                    'columns': data.columns.tolist(),
                    'dtypes': data.dtypes.to_dict()
                })
        except Exception:
            pass
        
        return info
