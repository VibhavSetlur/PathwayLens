"""
GMT (Gene Matrix Transposed) file format support for PathwayLens.

GMT format is commonly used for gene set files (e.g., MSigDB).
Format: pathway_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
"""

from pathlib import Path
from typing import Dict, List, Optional, Set
import gzip

from loguru import logger


class GMTIO:
    """Read and write GMT (Gene Matrix Transposed) files."""

    def __init__(self):
        self.logger = logger.bind(module="gmt_io")

    def read_gmt(
        self, 
        file_path: Path, 
        min_genes: int = 1,
        max_genes: Optional[int] = None
    ) -> Dict[str, Dict[str, any]]:
        """
        Read a GMT file and return gene sets.
        
        Args:
            file_path: Path to GMT file
            min_genes: Minimum number of genes in a gene set (default: 1)
            max_genes: Maximum number of genes in a gene set (default: None)
            
        Returns:
            Dictionary mapping pathway_id to pathway information:
            {
                "pathway_id": {
                    "name": str,
                    "description": str,
                    "genes": List[str],
                    "size": int
                }
            }
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"GMT file not found: {file_path}")
        
        self.logger.info(f"Reading GMT file: {file_path}")
        
        gene_sets = {}
        
        # Determine if file is gzipped
        open_func = gzip.open if file_path.suffix == ".gz" else open
        mode = "rt" if file_path.suffix == ".gz" else "r"
        
        with open_func(file_path, mode) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 3:
                    self.logger.warning(f"Line {line_num} has fewer than 3 tab-separated fields, skipping")
                    continue
                
                # Parse GMT format: name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
                pathway_name = parts[0].strip()
                description = parts[1].strip()
                genes = [g.strip() for g in parts[2:] if g.strip()]
                
                # Filter by gene count
                if len(genes) < min_genes:
                    continue
                if max_genes and len(genes) > max_genes:
                    continue
                
                # Use pathway name as ID if no ID specified
                pathway_id = pathway_name
                
                gene_sets[pathway_id] = {
                    "name": pathway_name,
                    "description": description,
                    "genes": genes,
                    "size": len(genes)
                }
        
        self.logger.info(f"Read {len(gene_sets)} gene sets from GMT file")
        return gene_sets

    def write_gmt(
        self,
        gene_sets: Dict[str, Dict[str, any]],
        file_path: Path,
        include_description: bool = True
    ) -> None:
        """
        Write gene sets to a GMT file.
        
        Args:
            gene_sets: Dictionary mapping pathway_id to pathway information
            file_path: Output file path
            include_description: Whether to include description column (default: True)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Writing {len(gene_sets)} gene sets to GMT file: {file_path}")
        
        # Determine if file should be gzipped
        open_func = gzip.open if file_path.suffix == ".gz" else open
        mode = "wt" if file_path.suffix == ".gz" else "w"
        
        with open_func(file_path, mode) as outfile:
            for pathway_id, pathway_info in gene_sets.items():
                name = pathway_info.get("name", pathway_id)
                description = pathway_info.get("description", "") if include_description else ""
                genes = pathway_info.get("genes", [])
                
                # Write GMT line: name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
                if include_description:
                    line = f"{name}\t{description}\t" + "\t".join(genes)
                else:
                    line = f"{name}\t\t" + "\t".join(genes)
                
                outfile.write(line + "\n")
        
        self.logger.info(f"Successfully wrote GMT file: {file_path}")

    def read_gmt_to_list(
        self,
        file_path: Path,
        min_genes: int = 1,
        max_genes: Optional[int] = None
    ) -> List[Dict[str, any]]:
        """
        Read GMT file and return as list of gene sets.
        
        Args:
            file_path: Path to GMT file
            min_genes: Minimum number of genes in a gene set
            max_genes: Maximum number of genes in a gene set
            
        Returns:
            List of gene set dictionaries
        """
        gene_sets_dict = self.read_gmt(file_path, min_genes, max_genes)
        return [
            {"pathway_id": pid, **info}
            for pid, info in gene_sets_dict.items()
        ]

    def filter_gene_sets(
        self,
        gene_sets: Dict[str, Dict[str, any]],
        min_genes: int = 1,
        max_genes: Optional[int] = None,
        gene_whitelist: Optional[Set[str]] = None,
        gene_blacklist: Optional[Set[str]] = None
    ) -> Dict[str, Dict[str, any]]:
        """
        Filter gene sets by size and gene membership.
        
        Args:
            gene_sets: Dictionary of gene sets
            min_genes: Minimum number of genes
            max_genes: Maximum number of genes
            gene_whitelist: Only include genes in this set (optional)
            gene_blacklist: Exclude genes in this set (optional)
            
        Returns:
            Filtered dictionary of gene sets
        """
        filtered = {}
        
        for pathway_id, pathway_info in gene_sets.items():
            genes = pathway_info.get("genes", [])
            
            # Apply gene filters
            if gene_whitelist:
                genes = [g for g in genes if g in gene_whitelist]
            if gene_blacklist:
                genes = [g for g in genes if g not in gene_blacklist]
            
            # Check size constraints
            if len(genes) < min_genes:
                continue
            if max_genes and len(genes) > max_genes:
                continue
            
            # Create filtered pathway info
            filtered[pathway_id] = {
                **pathway_info,
                "genes": genes,
                "size": len(genes)
            }
        
        return filtered

