"""
GCT (Gene Cluster Text) file format support for PathwayLens.

GCT format is commonly used for gene expression matrices.
Format: #1.2<TAB>rows<TAB>columns
        Name<TAB>Description<TAB>sample1<TAB>sample2<TAB>...
        gene1<TAB>desc1<TAB>value1<TAB>value2<TAB>...
"""

from pathlib import Path
from typing import Optional, Dict, List
import gzip
import pandas as pd

from loguru import logger


class GCTIO:
    """Read and write GCT (Gene Cluster Text) files."""

    def __init__(self):
        self.logger = logger.bind(module="gct_io")

    def read_gct(
        self,
        file_path: Path,
        gene_column: str = "Name",
        description_column: str = "Description",
        skip_rows: int = 2
    ) -> pd.DataFrame:
        """
        Read a GCT file and return as pandas DataFrame.
        
        Args:
            file_path: Path to GCT file
            gene_column: Name of the gene identifier column (default: "Name")
            description_column: Name of the description column (default: "Description")
            skip_rows: Number of header rows to skip (default: 2 for GCT format)
            
        Returns:
            DataFrame with genes as index and samples as columns
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"GCT file not found: {file_path}")
        
        self.logger.info(f"Reading GCT file: {file_path}")
        
        # Determine if file is gzipped
        open_func = gzip.open if file_path.suffix == ".gz" else open
        mode = "rt" if file_path.suffix == ".gz" else "r"
        
        # Read GCT format
        # Line 1: #1.2 or #1.3
        # Line 2: <rows><TAB><columns>
        # Line 3: Name<TAB>Description<TAB>sample1<TAB>sample2<TAB>...
        # Line 4+: gene<TAB>description<TAB>value1<TAB>value2<TAB>...
        
        with open_func(file_path, mode) as f:
            # Read version line
            version_line = f.readline().strip()
            if not version_line.startswith("#"):
                raise ValueError(f"Invalid GCT format: first line should start with #, got: {version_line}")
            
            # Read dimensions line
            dims_line = f.readline().strip()
            try:
                num_rows, num_cols = map(int, dims_line.split('\t')[:2])
            except ValueError:
                raise ValueError(f"Invalid GCT format: second line should be <rows><TAB><columns>, got: {dims_line}")
            
            # Read column headers
            header_line = f.readline().strip()
            if not header_line:
                raise ValueError("Invalid GCT format: missing header line")
            
            headers = header_line.split('\t')
            
            # Read data
            data = []
            row_names = []
            descriptions = []
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                row_name = parts[0]
                description = parts[1] if len(parts) > 1 else ""
                values = parts[2:]
                
                row_names.append(row_name)
                descriptions.append(description)
                data.append(values)
        
        # Create DataFrame
        # Column names are: gene_id, description, sample1, sample2, ...
        column_names = [gene_column, description_column] + headers[2:] if len(headers) > 2 else [gene_column, description_column]
        
        # Pad data if needed
        max_cols = max(len(row) for row in data) if data else 0
        column_names = column_names[:max_cols + 2]  # +2 for gene_id and description
        
        df_data = {}
        df_data[gene_column] = row_names
        df_data[description_column] = descriptions
        
        # Add sample columns
        num_samples = len(headers) - 2 if len(headers) > 2 else max_cols - 2
        for i in range(num_samples):
            col_name = headers[i + 2] if i + 2 < len(headers) else f"Sample_{i + 1}"
            df_data[col_name] = [row[i] if i < len(row) else None for row in data]
        
        df = pd.DataFrame(df_data)
        
        # Set gene column as index
        df.set_index(gene_column, inplace=True)
        
        # Convert numeric columns
        for col in df.columns:
            if col != description_column:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.logger.info(f"Read GCT file: {len(df)} rows, {len(df.columns)} columns")
        return df

    def write_gct(
        self,
        df: pd.DataFrame,
        file_path: Path,
        gene_column: Optional[str] = None,
        description_column: Optional[str] = None,
        version: str = "1.2"
    ) -> None:
        """
        Write DataFrame to GCT format.
        
        Args:
            df: DataFrame with genes as rows and samples as columns
            file_path: Output file path
            gene_column: Name of gene identifier column (if None, uses index)
            description_column: Name of description column (optional)
            version: GCT format version (default: "1.2")
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Writing DataFrame to GCT file: {file_path}")
        
        # Prepare data
        if gene_column and gene_column in df.columns:
            df = df.copy()
            df.index = df[gene_column]
            df = df.drop(columns=[gene_column])
        else:
            df = df.copy()
            df.index.name = "Name"
        
        # Reset index to get gene names as column
        df_with_index = df.reset_index()
        gene_col = df_with_index.columns[0]
        
        # Add description column if not present
        if description_column and description_column in df_with_index.columns:
            desc_col = description_column
        else:
            df_with_index["Description"] = ""
            desc_col = "Description"
        
        # Determine if file should be gzipped
        open_func = gzip.open if file_path.suffix == ".gz" else open
        mode = "wt" if file_path.suffix == ".gz" else "w"
        
        with open_func(file_path, mode) as outfile:
            # Write version line
            outfile.write(f"#{version}\n")
            
            # Write dimensions line
            num_rows = len(df_with_index)
            num_cols = len(df_with_index.columns) - 2  # Exclude gene and description columns
            outfile.write(f"{num_rows}\t{num_cols}\n")
            
            # Write header line
            sample_columns = [col for col in df_with_index.columns if col not in [gene_col, desc_col]]
            header = f"{gene_col}\t{desc_col}\t" + "\t".join(sample_columns) + "\n"
            outfile.write(header)
            
            # Write data lines
            for _, row in df_with_index.iterrows():
                gene = str(row[gene_col])
                desc = str(row[desc_col]) if desc_col in row else ""
                values = [str(row[col]) for col in sample_columns]
                line = f"{gene}\t{desc}\t" + "\t".join(values) + "\n"
                outfile.write(line)
        
        self.logger.info(f"Successfully wrote GCT file: {file_path}")

    def read_gct_simple(
        self,
        file_path: Path,
        gene_index: bool = True
    ) -> pd.DataFrame:
        """
        Read GCT file in a simpler format (returns just the expression matrix).
        
        Args:
            file_path: Path to GCT file
            gene_index: Whether to use genes as index (default: True)
            
        Returns:
            DataFrame with expression values
        """
        df = self.read_gct(file_path)
        
        # Remove description column if present
        if "Description" in df.columns:
            df = df.drop(columns=["Description"])
        
        return df

