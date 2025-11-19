"""
Property-based tests for statistical methods using Hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
import numpy as np
from typing import List

from pathwaylens_core.analysis.consensus_engine import ConsensusEngine
from pathwaylens_core.analysis.schemas import CorrectionMethod


@pytest.mark.property
class TestStatisticalProperties:
    """Property-based tests for statistical methods."""

    @pytest.fixture
    def consensus_engine(self):
        """Create consensus engine."""
        return ConsensusEngine()

    @given(
        p_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_fisher_method_properties(self, consensus_engine, p_values):
        """Test Fisher's method properties."""
        assume(len(p_values) > 0)
        assume(all(0 <= p <= 1 for p in p_values))
        
        # Fisher's method should produce a p-value between 0 and 1
        combined_p = consensus_engine._fisher_method(p_values)
        
        assert 0 <= combined_p <= 1, \
            f"Fisher's method produced invalid p-value: {combined_p}"
        
        # If all p-values are very small, combined should be very small
        if all(p < 0.001 for p in p_values):
            assert combined_p < 0.01, \
                "Fisher's method should produce small p-value for small inputs"
        
        # If all p-values are large, combined should be large
        if all(p > 0.5 for p in p_values):
            assert combined_p > 0.1, \
                "Fisher's method should produce large p-value for large inputs"

    @given(
        p_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_stouffer_method_properties(self, consensus_engine, p_values):
        """Test Stouffer's method properties."""
        assume(len(p_values) > 0)
        assume(all(0 <= p <= 1 for p in p_values))
        
        # Stouffer's method should produce a p-value between 0 and 1
        combined_p = consensus_engine._stouffer_method(p_values)
        
        assert 0 <= combined_p <= 1, \
            f"Stouffer's method produced invalid p-value: {combined_p}"
        
        # Commutativity: order shouldn't matter
        reversed_p = consensus_engine._stouffer_method(list(reversed(p_values)))
        assert abs(combined_p - reversed_p) < 1e-10, \
            "Stouffer's method should be commutative"

    @given(
        p_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_geometric_mean_properties(self, consensus_engine, p_values):
        """Test geometric mean method properties."""
        assume(len(p_values) > 0)
        assume(all(0 < p <= 1 for p in p_values))  # Geometric mean requires positive values
        
        # Geometric mean should produce a p-value between 0 and 1
        combined_p = consensus_engine._geometric_mean_method(p_values)
        
        assert 0 <= combined_p <= 1, \
            f"Geometric mean method produced invalid p-value: {combined_p}"
        
        # Geometric mean should be less than or equal to arithmetic mean
        arithmetic_mean = np.mean(p_values)
        assert combined_p <= arithmetic_mean, \
            "Geometric mean should be <= arithmetic mean"

    @given(
        p_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50
        ),
        r=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=30, deadline=5000)
    def test_wilkinson_method_properties(self, consensus_engine, p_values, r):
        """Test Wilkinson's method properties."""
        assume(len(p_values) >= r)
        assume(all(0 <= p <= 1 for p in p_values))
        
        # Wilkinson's method should produce a p-value between 0 and 1
        combined_p = consensus_engine._wilkinson_method(p_values, r=min(r, len(p_values)))
        
        assert 0 <= combined_p <= 1, \
            f"Wilkinson's method produced invalid p-value: {combined_p}"

    @given(
        p_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=50
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_pearson_method_properties(self, consensus_engine, p_values):
        """Test Pearson's method properties."""
        assume(len(p_values) > 0)
        assume(all(0 < p <= 1 for p in p_values))  # Product requires positive values
        
        # Pearson's method should produce a p-value between 0 and 1
        combined_p = consensus_engine._pearson_method(p_values)
        
        assert 0 <= combined_p <= 1, \
            f"Pearson's method produced invalid p-value: {combined_p}"

    @given(
        p_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_benjamini_hochberg_properties(self, consensus_engine, p_values):
        """Test Benjamini-Hochberg correction properties."""
        assume(len(p_values) > 0)
        assume(all(0 <= p <= 1 for p in p_values))
        
        # BH correction should produce adjusted p-values
        adjusted = consensus_engine._benjamini_hochberg(p_values)
        
        assert len(adjusted) == len(p_values), \
            "BH correction should return same number of p-values"
        
        assert all(0 <= p <= 1 for p in adjusted), \
            "BH correction should produce valid p-values"
        
        # Adjusted p-values should be >= original p-values
        for orig, adj in zip(p_values, adjusted):
            assert adj >= orig, \
                f"Adjusted p-value ({adj}) should be >= original ({orig})"
        
        # Adjusted p-values should be monotonic (sorted)
        assert all(adjusted[i] <= adjusted[i+1] for i in range(len(adjusted)-1)), \
            "BH adjusted p-values should be monotonic"

    @given(
        p_values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=50, deadline=5000)
    def test_bonferroni_properties(self, consensus_engine, p_values):
        """Test Bonferroni correction properties."""
        assume(len(p_values) > 0)
        assume(all(0 <= p <= 1 for p in p_values))
        
        # Bonferroni correction should produce adjusted p-values
        adjusted = consensus_engine._bonferroni(p_values)
        
        assert len(adjusted) == len(p_values), \
            "Bonferroni correction should return same number of p-values"
        
        assert all(0 <= p <= 1 for p in adjusted), \
            "Bonferroni correction should produce valid p-values"
        
        # Adjusted p-values should be >= original p-values
        for orig, adj in zip(p_values, adjusted):
            assert adj >= orig, \
                f"Adjusted p-value ({adj}) should be >= original ({orig})"
        
        # Bonferroni is more conservative than BH
        bh_adjusted = consensus_engine._benjamini_hochberg(p_values)
        for bonf, bh in zip(adjusted, bh_adjusted):
            assert bonf >= bh, \
                "Bonferroni should be more conservative than BH"



