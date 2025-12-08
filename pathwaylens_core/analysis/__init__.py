"""
Analysis module for PathwayLens.

This module provides pathway analysis capabilities including ORA, GSEA,
GSVA, network-based methods, Bayesian methods, and other enrichment analysis methods.
"""

from .engine import AnalysisEngine
from .ora_engine import ORAEngine
from .gsea_engine import GSEAEngine
from .gsva_engine import GSVAEngine
from .topology_engine import TopologyEngine
from .multi_omics_engine import MultiOmicsEngine
from .consensus_engine import ConsensusEngine
from .network_engine import NetworkEngine
from .bayesian_engine import BayesianEngine
from .pathway_interaction_engine import PathwayInteractionEngine, PathwayInteraction, InteractionAnalysisResult
from .schemas import AnalysisResult, PathwayResult, AnalysisParameters

from .sc_engine import SingleCellEngine, SingleCellResult

__all__ = [
    "AnalysisEngine",
    "ORAEngine",
    "GSEAEngine", 
    "GSVAEngine",
    "TopologyEngine",
    "MultiOmicsEngine",
    "ConsensusEngine",
    "NetworkEngine",
    "BayesianEngine",
    "PathwayInteractionEngine",
    "PathwayInteraction",
    "InteractionAnalysisResult",
    "AnalysisResult",
    "PathwayResult",
    "AnalysisParameters",
]
