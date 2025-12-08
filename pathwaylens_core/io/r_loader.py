"""
Loader for R objects and AnnData files.
Facilitates interoperability with R/Bioconductor and Scanpy ecosystems.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Any
from loguru import logger

class RLoader:
    """
    Loads R data files (.rds) and AnnData files (.h5ad).
    """
    
    @staticmethod
    def load_rds(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Load an R .rds file into a pandas DataFrame.
        Requires rpy2 to be installed.
        
        Args:
            file_path: Path to the .rds file.
            
        Returns:
            Pandas DataFrame or None if loading fails.
        """
        try:
            import rpy2.robjects as robjects
            from rpy2.robjects import pandas2ri
            
            # Activate pandas conversion
            pandas2ri.activate()
            
            path = str(file_path)
            logger.info(f"Loading R object from {path}")
            
            r_obj = robjects.r['readRDS'](path)
            
            # Convert to pandas DataFrame
            # This handles standard R data.frames and matrices
            # For complex objects like Seurat, we might need specific extraction logic
            # Here we attempt a generic conversion
            df = pandas2ri.rpy2py(r_obj)
            
            if isinstance(df, pd.DataFrame):
                return df
            else:
                logger.warning(f"Loaded object is not a DataFrame, but {type(df)}")
                # Try to convert to DataFrame if it's a matrix-like object
                try:
                    return pd.DataFrame(df)
                except:
                    return None
                    
        except ImportError:
            logger.error("rpy2 is not installed. Cannot load .rds files.")
            logger.info("Please install rpy2: pip install rpy2")
            return None
        except Exception as e:
            logger.error(f"Failed to load .rds file: {e}")
            return None

    @staticmethod
    def load_h5ad(file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Load an AnnData .h5ad file.
        Returns the expression matrix (X) as a DataFrame.
        
        Args:
            file_path: Path to the .h5ad file.
            
        Returns:
            Pandas DataFrame (cells x genes) or None.
        """
        try:
            import anndata
            import scipy.sparse
            
            path = str(file_path)
            logger.info(f"Loading AnnData from {path}")
            
            adata = anndata.read_h5ad(path)
            
            # Extract expression matrix
            if scipy.sparse.issparse(adata.X):
                data = adata.X.toarray()
            else:
                data = adata.X
                
            # Create DataFrame
            df = pd.DataFrame(
                data,
                index=adata.obs_names,
                columns=adata.var_names
            )
            
            return df
            
        except ImportError:
            logger.error("anndata is not installed. Cannot load .h5ad files.")
            logger.info("Please install anndata: pip install anndata")
            return None
        except Exception as e:
            logger.error(f"Failed to load .h5ad file: {e}")
            return None
