"""
Confidence scoring for gene ID normalization.

Calculates probabilistic confidence scores for ID mappings based on
multiple factors including database agreement, mapping frequency, etc.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ConfidenceFactors:
    """Factors contributing to confidence score."""
    database_agreement: float = 0.0
    mapping_frequency: float = 0.0
    database_quality: float = 0.0
    species_match: float = 1.0
    id_type_compatibility: float = 1.0


class ConfidenceCalculator:
    """Calculate confidence scores for ID mappings."""
    
    def __init__(self):
        """Initialize confidence calculator."""
        self.logger = logger.bind(module="confidence_calculator")
        
        # Database quality weights (higher = more reliable)
        self.database_weights = {
            'ensembl': 0.95,
            'mygene': 0.90,
            'ncbi': 0.85,
            'uniprot': 0.90,
            'hgnc': 0.95,
            'biomart': 0.85,
        }
    
    def calculate_confidence(
        self,
        mappings: List[Dict[str, any]],
        factors: Optional[ConfidenceFactors] = None
    ) -> float:
        """
        Calculate overall confidence score for a mapping.
        
        Args:
            mappings: List of mapping results from different databases
            factors: Optional confidence factors
            
        Returns:
            Confidence score between 0 and 1
        """
        if not mappings:
            return 0.0
        
        if factors is None:
            factors = self._extract_factors(mappings)
        
        # Enhanced weighted combination with adaptive weights
        # Weights adjust based on number of mappings available
        n_mappings = len(mappings)
        
        if n_mappings >= 3:
            # Multiple sources: emphasize agreement
            weights = {
                'agreement': 0.5,
                'frequency': 0.15,
                'quality': 0.2,
                'species': 0.1,
                'compatibility': 0.05
            }
        elif n_mappings == 2:
            # Two sources: balance agreement and quality
            weights = {
                'agreement': 0.4,
                'frequency': 0.2,
                'quality': 0.25,
                'species': 0.1,
                'compatibility': 0.05
            }
        else:
            # Single source: emphasize quality
            weights = {
                'agreement': 0.2,
                'frequency': 0.1,
                'quality': 0.5,
                'species': 0.1,
                'compatibility': 0.1
            }
        
        confidence = (
            weights['agreement'] * factors.database_agreement +
            weights['frequency'] * factors.mapping_frequency +
            weights['quality'] * factors.database_quality +
            weights['species'] * factors.species_match +
            weights['compatibility'] * factors.id_type_compatibility
        )
        
        # Apply non-linear transformation for better discrimination
        # Higher confidence gets boosted, lower gets penalized
        if confidence > 0.7:
            confidence = 0.7 + 0.3 * ((confidence - 0.7) / 0.3) ** 0.8
        elif confidence < 0.3:
            confidence = 0.3 * (confidence / 0.3) ** 1.2
        
        return min(max(confidence, 0.0), 1.0)
    
    def _extract_factors(
        self,
        mappings: List[Dict[str, any]]
    ) -> ConfidenceFactors:
        """Extract confidence factors from mappings."""
        factors = ConfidenceFactors()
        
        if not mappings:
            return factors
        
        # Database agreement: how many databases agree
        unique_outputs = set(m.get('output_id') for m in mappings if m.get('output_id'))
        if len(unique_outputs) == 1:
            factors.database_agreement = 1.0
        elif len(unique_outputs) == 2:
            factors.database_agreement = 0.7
        else:
            factors.database_agreement = 0.3
        
        # Mapping frequency: how often this mapping appears
        if mappings:
            factors.mapping_frequency = min(1.0, len(mappings) / 3.0)
        
        # Database quality: average quality of databases used
        db_qualities = [
            self.database_weights.get(m.get('source', '').lower(), 0.5)
            for m in mappings
        ]
        if db_qualities:
            factors.database_quality = np.mean(db_qualities)
        
        # Species match (assumed 1.0 if not specified)
        factors.species_match = 1.0
        
        # ID type compatibility (assumed 1.0 if not specified)
        factors.id_type_compatibility = 1.0
        
        return factors
    
    def calculate_consensus_mapping(
        self,
        mappings: List[Dict[str, any]],
        min_confidence: float = 0.5
    ) -> Optional[Dict[str, any]]:
        """
        Calculate consensus mapping from multiple database results.
        
        Args:
            mappings: List of mapping results
            min_confidence: Minimum confidence threshold
            
        Returns:
            Consensus mapping or None if confidence too low
        """
        if not mappings:
            return None
        
        # Group by output ID
        output_groups = {}
        for mapping in mappings:
            output_id = mapping.get('output_id')
            if output_id:
                if output_id not in output_groups:
                    output_groups[output_id] = []
                output_groups[output_id].append(mapping)
        
        if not output_groups:
            return None
        
        # Find most common output ID
        best_output = max(
            output_groups.keys(),
            key=lambda x: len(output_groups[x])
        )
        
        # Calculate confidence for this mapping
        best_mappings = output_groups[best_output]
        confidence = self.calculate_confidence(best_mappings)
        
        if confidence < min_confidence:
            return None
        
        # Return consensus mapping
        consensus = {
            'input_id': mappings[0].get('input_id'),
            'output_id': best_output,
            'confidence': confidence,
            'source': 'consensus',
            'alternative_mappings': [
                out_id for out_id in output_groups.keys()
                if out_id != best_output
            ]
        }
        
        return consensus
    
    def resolve_ambiguity(
        self,
        ambiguous_mappings: List[Dict[str, any]],
        context: Optional[Dict[str, any]] = None
    ) -> Optional[Dict[str, any]]:
        """
        Resolve ambiguous mappings using context.
        
        Args:
            ambiguous_mappings: List of ambiguous mappings
            context: Optional context information (e.g., pathway, tissue)
            
        Returns:
            Resolved mapping or None
        """
        if not ambiguous_mappings:
            return None
        
        # If context provided, use it to disambiguate
        if context:
            # Would use context to filter/rank mappings
            # For now, use highest confidence
            best = max(
                ambiguous_mappings,
                key=lambda x: x.get('confidence', 0.0)
            )
            return best
        
        # Otherwise, use consensus
        return self.calculate_consensus_mapping(ambiguous_mappings)

