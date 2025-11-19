"""
Unified identifier mapping for genes, proteins, and metabolites with caching.
"""

from .mapper import (
    map_gene_ids,
    map_protein_ids,
    map_metabolite_ids,
    MappingAudit,
)

__all__ = [
    "map_gene_ids",
    "map_protein_ids",
    "map_metabolite_ids",
    "MappingAudit",
]












