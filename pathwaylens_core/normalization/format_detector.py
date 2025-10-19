"""
Format detection for various input file types.
"""

import csv
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import chardet
from loguru import logger

from .schemas import FormatDetectionResult


class FormatDetector:
    """Detects file format and structure for various input types."""
    
    SUPPORTED_FORMATS = {
        'csv': ['.csv'],
        'tsv': ['.tsv', '.txt'],
        'excel': ['.xlsx', '.xls'],
        'json': ['.json'],
        'hdf5': ['.h5', '.hdf5'],
        'parquet': ['.parquet'],
        'feather': ['.feather'],
    }
    
    COMMON_DELIMITERS = [',', '\t', ';', '|', ' ']
    
    def __init__(self):
        self.logger = logger.bind(module="format_detector")
    
    def detect_format(self, file_path: str) -> FormatDetectionResult:
        """
        Detect the format of an input file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            FormatDetectionResult with detected format information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect by extension first
        extension = file_path.suffix.lower()
        format_type = self._detect_by_extension(extension)
        
        if format_type:
            return self._analyze_format(file_path, format_type)
        
        # Try to detect by content
        return self._detect_by_content(file_path)
    
    def _detect_by_extension(self, extension: str) -> Optional[str]:
        """Detect format by file extension."""
        for format_type, extensions in self.SUPPORTED_FORMATS.items():
            if extension in extensions:
                return format_type
        return None
    
    def _detect_by_content(self, file_path: Path) -> FormatDetectionResult:
        """Detect format by analyzing file content."""
        try:
            # Try to read as text first
            encoding = self._detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline().strip()
            
            # Try JSON
            if self._is_json(first_line):
                return self._analyze_format(file_path, 'json')
            
            # Try CSV/TSV
            delimiter = self._detect_delimiter(first_line)
            if delimiter:
                format_type = 'tsv' if delimiter == '\t' else 'csv'
                return self._analyze_format(file_path, format_type)
            
            # Default to CSV
            return self._analyze_format(file_path, 'csv')
            
        except Exception as e:
            self.logger.error(f"Error detecting format by content: {e}")
            return FormatDetectionResult(
                format_type='unknown',
                confidence=0.0,
                errors=[f"Could not detect format: {e}"]
            )
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get('encoding', 'utf-8')
        except Exception:
            return 'utf-8'
    
    def _is_json(self, line: str) -> bool:
        """Check if line is valid JSON."""
        try:
            json.loads(line)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def _detect_delimiter(self, line: str) -> Optional[str]:
        """Detect delimiter in a line of text."""
        delimiter_counts = {}
        
        for delimiter in self.COMMON_DELIMITERS:
            count = line.count(delimiter)
            if count > 0:
                delimiter_counts[delimiter] = count
        
        if not delimiter_counts:
            return None
        
        # Return delimiter with highest count
        return max(delimiter_counts, key=delimiter_counts.get)
    
    def _analyze_format(self, file_path: Path, format_type: str) -> FormatDetectionResult:
        """Analyze a file of known format type."""
        try:
            if format_type == 'csv':
                return self._analyze_csv(file_path)
            elif format_type == 'tsv':
                return self._analyze_tsv(file_path)
            elif format_type == 'excel':
                return self._analyze_excel(file_path)
            elif format_type == 'json':
                return self._analyze_json(file_path)
            elif format_type == 'hdf5':
                return self._analyze_hdf5(file_path)
            elif format_type == 'parquet':
                return self._analyze_parquet(file_path)
            elif format_type == 'feather':
                return self._analyze_feather(file_path)
            else:
                return FormatDetectionResult(
                    format_type=format_type,
                    confidence=0.0,
                    errors=[f"Unsupported format: {format_type}"]
                )
        except Exception as e:
            self.logger.error(f"Error analyzing {format_type} file: {e}")
            return FormatDetectionResult(
                format_type=format_type,
                confidence=0.0,
                errors=[f"Error analyzing file: {e}"]
            )
    
    def _analyze_csv(self, file_path: Path) -> FormatDetectionResult:
        """Analyze CSV file."""
        try:
            # Try to detect delimiter
            encoding = self._detect_encoding(file_path)
            delimiter = None
            
            with open(file_path, 'r', encoding=encoding) as f:
                sample = f.read(1024)
                delimiter = self._detect_delimiter(sample)
            
            if not delimiter:
                delimiter = ','
            
            # Read sample data
            df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, nrows=5)
            
            return FormatDetectionResult(
                format_type='csv',
                confidence=0.9,
                delimiter=delimiter,
                encoding=encoding,
                has_header=True,
                column_mapping={col: col for col in df.columns},
                sample_data=df.head(3).to_dict('records')
            )
        except Exception as e:
            return FormatDetectionResult(
                format_type='csv',
                confidence=0.5,
                errors=[f"CSV analysis error: {e}"]
            )
    
    def _analyze_tsv(self, file_path: Path) -> FormatDetectionResult:
        """Analyze TSV file."""
        try:
            encoding = self._detect_encoding(file_path)
            df = pd.read_csv(file_path, delimiter='\t', encoding=encoding, nrows=5)
            
            return FormatDetectionResult(
                format_type='tsv',
                confidence=0.9,
                delimiter='\t',
                encoding=encoding,
                has_header=True,
                column_mapping={col: col for col in df.columns},
                sample_data=df.head(3).to_dict('records')
            )
        except Exception as e:
            return FormatDetectionResult(
                format_type='tsv',
                confidence=0.5,
                errors=[f"TSV analysis error: {e}"]
            )
    
    def _analyze_excel(self, file_path: Path) -> FormatDetectionResult:
        """Analyze Excel file."""
        try:
            # Read first sheet
            df = pd.read_excel(file_path, nrows=5)
            
            return FormatDetectionResult(
                format_type='excel',
                confidence=0.9,
                encoding='utf-8',
                has_header=True,
                column_mapping={col: col for col in df.columns},
                sample_data=df.head(3).to_dict('records')
            )
        except Exception as e:
            return FormatDetectionResult(
                format_type='excel',
                confidence=0.5,
                errors=[f"Excel analysis error: {e}"]
            )
    
    def _analyze_json(self, file_path: Path) -> FormatDetectionResult:
        """Analyze JSON file."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Determine if it's a list or dict
            if isinstance(data, list) and len(data) > 0:
                sample_data = data[:3]
                if isinstance(data[0], dict):
                    column_mapping = {key: key for key in data[0].keys()}
                else:
                    column_mapping = {}
            elif isinstance(data, dict):
                sample_data = [data]
                column_mapping = {key: key for key in data.keys()}
            else:
                sample_data = []
                column_mapping = {}
            
            return FormatDetectionResult(
                format_type='json',
                confidence=0.9,
                encoding='utf-8',
                has_header=False,
                column_mapping=column_mapping,
                sample_data=sample_data
            )
        except Exception as e:
            return FormatDetectionResult(
                format_type='json',
                confidence=0.5,
                errors=[f"JSON analysis error: {e}"]
            )
    
    def _analyze_hdf5(self, file_path: Path) -> FormatDetectionResult:
        """Analyze HDF5 file."""
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                keys = list(f.keys())
                if keys:
                    # Get sample data from first dataset
                    dataset = f[keys[0]]
                    if hasattr(dataset, 'shape') and len(dataset.shape) > 0:
                        sample_size = min(3, dataset.shape[0])
                        sample_data = dataset[:sample_size].tolist()
                    else:
                        sample_data = []
                else:
                    sample_data = []
            
            return FormatDetectionResult(
                format_type='hdf5',
                confidence=0.9,
                encoding='utf-8',
                has_header=False,
                column_mapping={},
                sample_data=sample_data
            )
        except Exception as e:
            return FormatDetectionResult(
                format_type='hdf5',
                confidence=0.5,
                errors=[f"HDF5 analysis error: {e}"]
            )
    
    def _analyze_parquet(self, file_path: Path) -> FormatDetectionResult:
        """Analyze Parquet file."""
        try:
            df = pd.read_parquet(file_path, nrows=5)
            
            return FormatDetectionResult(
                format_type='parquet',
                confidence=0.9,
                encoding='utf-8',
                has_header=True,
                column_mapping={col: col for col in df.columns},
                sample_data=df.head(3).to_dict('records')
            )
        except Exception as e:
            return FormatDetectionResult(
                format_type='parquet',
                confidence=0.5,
                errors=[f"Parquet analysis error: {e}"]
            )
    
    def _analyze_feather(self, file_path: Path) -> FormatDetectionResult:
        """Analyze Feather file."""
        try:
            df = pd.read_feather(file_path)
            df_sample = df.head(5)
            
            return FormatDetectionResult(
                format_type='feather',
                confidence=0.9,
                encoding='utf-8',
                has_header=True,
                column_mapping={col: col for col in df.columns},
                sample_data=df_sample.head(3).to_dict('records')
            )
        except Exception as e:
            return FormatDetectionResult(
                format_type='feather',
                confidence=0.5,
                errors=[f"Feather analysis error: {e}"]
            )
