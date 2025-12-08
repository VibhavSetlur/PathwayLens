"""
Database adapters for PathwayLens.

This module provides adapters for various pathway and gene databases,
enabling unified access to different data sources.
"""

from .base import BaseAdapter
from .kegg_adapter import KEGGAdapter
from .reactome_adapter import ReactomeAdapter
from .go_adapter import GOAdapter
from .biocyc_adapter import BioCycAdapter
from .pathway_commons_adapter import PathwayCommonsAdapter
from .msigdb_adapter import MSigDBAdapter
from .panther_adapter import PantherAdapter
from .wikipathways_adapter import WikiPathwaysAdapter
from .custom_adapter import CustomAdapter

__all__ = [
    "BaseAdapter",
    "KEGGAdapter",
    "ReactomeAdapter",
    "GOAdapter", 
    "BioCycAdapter",
    "PathwayCommonsAdapter",
    "MSigDBAdapter",
    "PantherAdapter",
    "WikiPathwaysAdapter",
    "CustomAdapter",
]
