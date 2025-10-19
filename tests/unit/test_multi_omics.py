"""
Unit tests for the Multi-omics modules.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.multi_omics.proteomics import ProteomicsAnalyzer
from pathwaylens_core.multi_omics.metabolomics import MetabolomicsAnalyzer
from pathwaylens_core.multi_omics.phosphoproteomics import PhosphoproteomicsAnalyzer
from pathwaylens_core.multi_omics.epigenomics import EpigenomicsAnalyzer
from pathwaylens_core.multi_omics.joint_analysis import JointMultiOmicsAnalyzer
from pathwaylens_core.multi_omics.time_course import TimeCourseAnalyzer


class TestProteomicsAnalyzer:
    """Test cases for the ProteomicsAnalyzer class."""

    @pytest.fixture
    def proteomics_analyzer(self):
        """Create a ProteomicsAnalyzer instance for testing."""
        return ProteomicsAnalyzer()

    @pytest.fixture
    def sample_protein_ids(self):
        """Create sample protein IDs."""
        return ["P12345", "P67890", "P11111", "P22222", "P33333"]

    @pytest.fixture
    def sample_expression_data(self):
        """Create sample expression data."""
        return pd.DataFrame({
            "P12345": [1.0, 2.0, 3.0, 4.0, 5.0],
            "P67890": [2.0, 4.0, 6.0, 8.0, 10.0],
            "P11111": [0.5, 1.0, 1.5, 2.0, 2.5],
            "P22222": [3.0, 6.0, 9.0, 12.0, 15.0],
            "P33333": [1.5, 3.0, 4.5, 6.0, 7.5]
        })

    def test_init(self, proteomics_analyzer):
        """Test ProteomicsAnalyzer initialization."""
        assert proteomics_analyzer.logger is not None
        assert proteomics_analyzer.uniprot_rest_url is not None

    @pytest.mark.asyncio
    async def test_analyze_proteomics_data_basic(self, proteomics_analyzer, sample_protein_ids):
        """Test basic proteomics data analysis."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "results": [
                    {"primaryAccession": "P12345", "proteinName": "Test Protein 1"},
                    {"primaryAccession": "P67890", "proteinName": "Test Protein 2"}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await proteomics_analyzer.analyze_proteomics_data(
                protein_ids=sample_protein_ids,
                species="human"
            )

            assert isinstance(result, dict)
            assert "protein_data" in result
            assert "analysis_summary" in result
            assert len(result["protein_data"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_proteomics_data_with_expression(self, proteomics_analyzer, sample_protein_ids, sample_expression_data):
        """Test proteomics data analysis with expression data."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "results": [
                    {"primaryAccession": "P12345", "proteinName": "Test Protein 1"},
                    {"primaryAccession": "P67890", "proteinName": "Test Protein 2"}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await proteomics_analyzer.analyze_proteomics_data(
                protein_ids=sample_protein_ids,
                species="human",
                expression_data=sample_expression_data
            )

            assert isinstance(result, dict)
            assert "protein_data" in result
            assert "expression_analysis" in result
            assert "analysis_summary" in result

    @pytest.mark.asyncio
    async def test_analyze_proteomics_data_empty_input(self, proteomics_analyzer):
        """Test proteomics data analysis with empty input."""
        result = await proteomics_analyzer.analyze_proteomics_data(
            protein_ids=[],
            species="human"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "Empty protein ID list" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_proteomics_data_api_error(self, proteomics_analyzer, sample_protein_ids):
        """Test proteomics data analysis with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await proteomics_analyzer.analyze_proteomics_data(
                protein_ids=sample_protein_ids,
                species="human"
            )

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_protein_ids(self, proteomics_analyzer):
        """Test protein ID validation."""
        # Valid protein IDs
        assert proteomics_analyzer._validate_protein_ids(["P12345", "P67890"]) is True
        
        # Invalid protein IDs
        assert proteomics_analyzer._validate_protein_ids([]) is False
        assert proteomics_analyzer._validate_protein_ids(None) is False

    def test_validate_species(self, proteomics_analyzer):
        """Test species validation."""
        # Valid species
        assert proteomics_analyzer._validate_species("human") is True
        assert proteomics_analyzer._validate_species("mouse") is True
        
        # Invalid species
        assert proteomics_analyzer._validate_species("") is False
        assert proteomics_analyzer._validate_species(None) is False

    def test_validate_expression_data(self, proteomics_analyzer):
        """Test expression data validation."""
        # Valid expression data
        df = pd.DataFrame({"P12345": [1, 2, 3]})
        assert proteomics_analyzer._validate_expression_data(df) is True
        
        # Invalid expression data
        assert proteomics_analyzer._validate_expression_data(None) is False
        assert proteomics_analyzer._validate_expression_data(pd.DataFrame()) is False


class TestMetabolomicsAnalyzer:
    """Test cases for the MetabolomicsAnalyzer class."""

    @pytest.fixture
    def metabolomics_analyzer(self):
        """Create a MetabolomicsAnalyzer instance for testing."""
        return MetabolomicsAnalyzer()

    @pytest.fixture
    def sample_metabolite_ids(self):
        """Create sample metabolite IDs."""
        return ["HMDB0000001", "HMDB0000002", "HMDB0000003", "HMDB0000004", "HMDB0000005"]

    @pytest.fixture
    def sample_concentration_data(self):
        """Create sample concentration data."""
        return pd.DataFrame({
            "HMDB0000001": [10.0, 20.0, 30.0, 40.0, 50.0],
            "HMDB0000002": [5.0, 10.0, 15.0, 20.0, 25.0],
            "HMDB0000003": [2.0, 4.0, 6.0, 8.0, 10.0],
            "HMDB0000004": [15.0, 30.0, 45.0, 60.0, 75.0],
            "HMDB0000005": [1.0, 2.0, 3.0, 4.0, 5.0]
        })

    def test_init(self, metabolomics_analyzer):
        """Test MetabolomicsAnalyzer initialization."""
        assert metabolomics_analyzer.logger is not None
        assert metabolomics_analyzer.hmdb_api_url is not None

    @pytest.mark.asyncio
    async def test_analyze_metabolomics_data_basic(self, metabolomics_analyzer, sample_metabolite_ids):
        """Test basic metabolomics data analysis."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "metabolites": [
                    {"accession": "HMDB0000001", "name": "Test Metabolite 1"},
                    {"accession": "HMDB0000002", "name": "Test Metabolite 2"}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await metabolomics_analyzer.analyze_metabolomics_data(
                metabolite_ids=sample_metabolite_ids,
                species="human"
            )

            assert isinstance(result, dict)
            assert "metabolite_data" in result
            assert "analysis_summary" in result
            assert len(result["metabolite_data"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_metabolomics_data_with_concentration(self, metabolomics_analyzer, sample_metabolite_ids, sample_concentration_data):
        """Test metabolomics data analysis with concentration data."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "metabolites": [
                    {"accession": "HMDB0000001", "name": "Test Metabolite 1"},
                    {"accession": "HMDB0000002", "name": "Test Metabolite 2"}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await metabolomics_analyzer.analyze_metabolomics_data(
                metabolite_ids=sample_metabolite_ids,
                species="human",
                concentration_data=sample_concentration_data
            )

            assert isinstance(result, dict)
            assert "metabolite_data" in result
            assert "concentration_analysis" in result
            assert "analysis_summary" in result

    @pytest.mark.asyncio
    async def test_analyze_metabolomics_data_empty_input(self, metabolomics_analyzer):
        """Test metabolomics data analysis with empty input."""
        result = await metabolomics_analyzer.analyze_metabolomics_data(
            metabolite_ids=[],
            species="human"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "Empty metabolite ID list" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_metabolomics_data_api_error(self, metabolomics_analyzer, sample_metabolite_ids):
        """Test metabolomics data analysis with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await metabolomics_analyzer.analyze_metabolomics_data(
                metabolite_ids=sample_metabolite_ids,
                species="human"
            )

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_metabolite_ids(self, metabolomics_analyzer):
        """Test metabolite ID validation."""
        # Valid metabolite IDs
        assert metabolomics_analyzer._validate_metabolite_ids(["HMDB0000001", "HMDB0000002"]) is True
        
        # Invalid metabolite IDs
        assert metabolomics_analyzer._validate_metabolite_ids([]) is False
        assert metabolomics_analyzer._validate_metabolite_ids(None) is False

    def test_validate_species(self, metabolomics_analyzer):
        """Test species validation."""
        # Valid species
        assert metabolomics_analyzer._validate_species("human") is True
        assert metabolomics_analyzer._validate_species("mouse") is True
        
        # Invalid species
        assert metabolomics_analyzer._validate_species("") is False
        assert metabolomics_analyzer._validate_species(None) is False

    def test_validate_concentration_data(self, metabolomics_analyzer):
        """Test concentration data validation."""
        # Valid concentration data
        df = pd.DataFrame({"HMDB0000001": [10, 20, 30]})
        assert metabolomics_analyzer._validate_concentration_data(df) is True
        
        # Invalid concentration data
        assert metabolomics_analyzer._validate_concentration_data(None) is False
        assert metabolomics_analyzer._validate_concentration_data(pd.DataFrame()) is False


class TestPhosphoproteomicsAnalyzer:
    """Test cases for the PhosphoproteomicsAnalyzer class."""

    @pytest.fixture
    def phosphoproteomics_analyzer(self):
        """Create a PhosphoproteomicsAnalyzer instance for testing."""
        return PhosphoproteomicsAnalyzer()

    @pytest.fixture
    def sample_phospho_sites(self):
        """Create sample phospho sites."""
        return ["P00519-S15", "P00519-T18", "P00519-Y20", "P12345-S10", "P12345-T25"]

    @pytest.fixture
    def sample_expression_data(self):
        """Create sample expression data."""
        return pd.DataFrame({
            "P00519-S15": [1.0, 2.0, 3.0, 4.0, 5.0],
            "P00519-T18": [2.0, 4.0, 6.0, 8.0, 10.0],
            "P00519-Y20": [0.5, 1.0, 1.5, 2.0, 2.5],
            "P12345-S10": [3.0, 6.0, 9.0, 12.0, 15.0],
            "P12345-T25": [1.5, 3.0, 4.5, 6.0, 7.5]
        })

    def test_init(self, phosphoproteomics_analyzer):
        """Test PhosphoproteomicsAnalyzer initialization."""
        assert phosphoproteomics_analyzer.logger is not None

    @pytest.mark.asyncio
    async def test_analyze_phosphoproteomics_data_basic(self, phosphoproteomics_analyzer, sample_phospho_sites):
        """Test basic phosphoproteomics data analysis."""
        result = await phosphoproteomics_analyzer.analyze_phosphoproteomics_data(
            phospho_sites=sample_phospho_sites,
            species="human"
        )

        assert isinstance(result, dict)
        assert "phospho_site_data" in result
        assert "analysis_summary" in result
        assert len(result["phospho_site_data"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_phosphoproteomics_data_with_expression(self, phosphoproteomics_analyzer, sample_phospho_sites, sample_expression_data):
        """Test phosphoproteomics data analysis with expression data."""
        result = await phosphoproteomics_analyzer.analyze_phosphoproteomics_data(
            phospho_sites=sample_phospho_sites,
            species="human",
            expression_data=sample_expression_data
        )

        assert isinstance(result, dict)
        assert "phospho_site_data" in result
        assert "expression_analysis" in result
        assert "analysis_summary" in result

    @pytest.mark.asyncio
    async def test_analyze_phosphoproteomics_data_empty_input(self, phosphoproteomics_analyzer):
        """Test phosphoproteomics data analysis with empty input."""
        result = await phosphoproteomics_analyzer.analyze_phosphoproteomics_data(
            phospho_sites=[],
            species="human"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "Empty phospho site list" in result["error"]

    def test_validate_phospho_sites(self, phosphoproteomics_analyzer):
        """Test phospho site validation."""
        # Valid phospho sites
        assert phosphoproteomics_analyzer._validate_phospho_sites(["P00519-S15", "P00519-T18"]) is True
        
        # Invalid phospho sites
        assert phosphoproteomics_analyzer._validate_phospho_sites([]) is False
        assert phosphoproteomics_analyzer._validate_phospho_sites(None) is False

    def test_validate_species(self, phosphoproteomics_analyzer):
        """Test species validation."""
        # Valid species
        assert phosphoproteomics_analyzer._validate_species("human") is True
        assert phosphoproteomics_analyzer._validate_species("mouse") is True
        
        # Invalid species
        assert phosphoproteomics_analyzer._validate_species("") is False
        assert phosphoproteomics_analyzer._validate_species(None) is False

    def test_validate_expression_data(self, phosphoproteomics_analyzer):
        """Test expression data validation."""
        # Valid expression data
        df = pd.DataFrame({"P00519-S15": [1, 2, 3]})
        assert phosphoproteomics_analyzer._validate_expression_data(df) is True
        
        # Invalid expression data
        assert phosphoproteomics_analyzer._validate_expression_data(None) is False
        assert phosphoproteomics_analyzer._validate_expression_data(pd.DataFrame()) is False


class TestEpigenomicsAnalyzer:
    """Test cases for the EpigenomicsAnalyzer class."""

    @pytest.fixture
    def epigenomics_analyzer(self):
        """Create an EpigenomicsAnalyzer instance for testing."""
        return EpigenomicsAnalyzer()

    @pytest.fixture
    def sample_genomic_regions(self):
        """Create sample genomic regions."""
        return ["chr1:100-200", "chr1:300-400", "chr2:500-600", "chr2:700-800", "chr3:900-1000"]

    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            "chr1:100-200": [1.0, 2.0, 3.0, 4.0, 5.0],
            "chr1:300-400": [2.0, 4.0, 6.0, 8.0, 10.0],
            "chr2:500-600": [0.5, 1.0, 1.5, 2.0, 2.5],
            "chr2:700-800": [3.0, 6.0, 9.0, 12.0, 15.0],
            "chr3:900-1000": [1.5, 3.0, 4.5, 6.0, 7.5]
        })

    def test_init(self, epigenomics_analyzer):
        """Test EpigenomicsAnalyzer initialization."""
        assert epigenomics_analyzer.logger is not None

    @pytest.mark.asyncio
    async def test_analyze_epigenomics_data_basic(self, epigenomics_analyzer, sample_genomic_regions):
        """Test basic epigenomics data analysis."""
        result = await epigenomics_analyzer.analyze_epigenomics_data(
            genomic_regions=sample_genomic_regions,
            species="human"
        )

        assert isinstance(result, dict)
        assert "genomic_region_data" in result
        assert "analysis_summary" in result
        assert len(result["genomic_region_data"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_epigenomics_data_with_data(self, epigenomics_analyzer, sample_genomic_regions, sample_data):
        """Test epigenomics data analysis with data."""
        result = await epigenomics_analyzer.analyze_epigenomics_data(
            genomic_regions=sample_genomic_regions,
            species="human",
            data=sample_data
        )

        assert isinstance(result, dict)
        assert "genomic_region_data" in result
        assert "data_analysis" in result
        assert "analysis_summary" in result

    @pytest.mark.asyncio
    async def test_analyze_epigenomics_data_empty_input(self, epigenomics_analyzer):
        """Test epigenomics data analysis with empty input."""
        result = await epigenomics_analyzer.analyze_epigenomics_data(
            genomic_regions=[],
            species="human"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "Empty genomic region list" in result["error"]

    def test_validate_genomic_regions(self, epigenomics_analyzer):
        """Test genomic region validation."""
        # Valid genomic regions
        assert epigenomics_analyzer._validate_genomic_regions(["chr1:100-200", "chr1:300-400"]) is True
        
        # Invalid genomic regions
        assert epigenomics_analyzer._validate_genomic_regions([]) is False
        assert epigenomics_analyzer._validate_genomic_regions(None) is False

    def test_validate_species(self, epigenomics_analyzer):
        """Test species validation."""
        # Valid species
        assert epigenomics_analyzer._validate_species("human") is True
        assert epigenomics_analyzer._validate_species("mouse") is True
        
        # Invalid species
        assert epigenomics_analyzer._validate_species("") is False
        assert epigenomics_analyzer._validate_species(None) is False

    def test_validate_data(self, epigenomics_analyzer):
        """Test data validation."""
        # Valid data
        df = pd.DataFrame({"chr1:100-200": [1, 2, 3]})
        assert epigenomics_analyzer._validate_data(df) is True
        
        # Invalid data
        assert epigenomics_analyzer._validate_data(None) is False
        assert epigenomics_analyzer._validate_data(pd.DataFrame()) is False


class TestJointMultiOmicsAnalyzer:
    """Test cases for the JointMultiOmicsAnalyzer class."""

    @pytest.fixture
    def joint_analyzer(self):
        """Create a JointMultiOmicsAnalyzer instance for testing."""
        return JointMultiOmicsAnalyzer()

    @pytest.fixture
    def sample_omics_data_results(self):
        """Create sample omics data results."""
        return {
            "proteomics": {"protein_data": [{"id": "P12345", "expression": 2.5}]},
            "transcriptomics": {"gene_data": [{"id": "GENE1", "expression": 1.8}]},
            "metabolomics": {"metabolite_data": [{"id": "HMDB0000001", "concentration": 15.0}]}
        }

    def test_init(self, joint_analyzer):
        """Test JointMultiOmicsAnalyzer initialization."""
        assert joint_analyzer.logger is not None

    @pytest.mark.asyncio
    async def test_perform_joint_analysis_basic(self, joint_analyzer, sample_omics_data_results):
        """Test basic joint multi-omics analysis."""
        result = await joint_analyzer.perform_joint_analysis(
            omics_data_results=sample_omics_data_results,
            analysis_type="pathway_activity_inference",
            species="human"
        )

        assert isinstance(result, dict)
        assert "joint_analysis_results" in result
        assert "analysis_summary" in result
        assert len(result["joint_analysis_results"]) > 0

    @pytest.mark.asyncio
    async def test_perform_joint_analysis_with_parameters(self, joint_analyzer, sample_omics_data_results):
        """Test joint multi-omics analysis with custom parameters."""
        parameters = {
            "threshold": 0.05,
            "method": "consensus",
            "min_omics": 2
        }

        result = await joint_analyzer.perform_joint_analysis(
            omics_data_results=sample_omics_data_results,
            analysis_type="pathway_activity_inference",
            species="human",
            parameters=parameters
        )

        assert isinstance(result, dict)
        assert "joint_analysis_results" in result
        assert "analysis_summary" in result

    @pytest.mark.asyncio
    async def test_perform_joint_analysis_empty_input(self, joint_analyzer):
        """Test joint multi-omics analysis with empty input."""
        result = await joint_analyzer.perform_joint_analysis(
            omics_data_results={},
            analysis_type="pathway_activity_inference",
            species="human"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "Empty omics data results" in result["error"]

    def test_validate_omics_data_results(self, joint_analyzer):
        """Test omics data results validation."""
        # Valid omics data results
        assert joint_analyzer._validate_omics_data_results({"proteomics": {}}) is True
        
        # Invalid omics data results
        assert joint_analyzer._validate_omics_data_results({}) is False
        assert joint_analyzer._validate_omics_data_results(None) is False

    def test_validate_analysis_type(self, joint_analyzer):
        """Test analysis type validation."""
        # Valid analysis types
        assert joint_analyzer._validate_analysis_type("pathway_activity_inference") is True
        assert joint_analyzer._validate_analysis_type("network_analysis") is True
        
        # Invalid analysis type
        assert joint_analyzer._validate_analysis_type("") is False
        assert joint_analyzer._validate_analysis_type(None) is False

    def test_validate_species(self, joint_analyzer):
        """Test species validation."""
        # Valid species
        assert joint_analyzer._validate_species("human") is True
        assert joint_analyzer._validate_species("mouse") is True
        
        # Invalid species
        assert joint_analyzer._validate_species("") is False
        assert joint_analyzer._validate_species(None) is False

    def test_validate_parameters(self, joint_analyzer):
        """Test parameters validation."""
        # Valid parameters
        assert joint_analyzer._validate_parameters({"threshold": 0.05}) is True
        
        # Invalid parameters
        assert joint_analyzer._validate_parameters(None) is False


class TestTimeCourseAnalyzer:
    """Test cases for the TimeCourseAnalyzer class."""

    @pytest.fixture
    def time_course_analyzer(self):
        """Create a TimeCourseAnalyzer instance for testing."""
        return TimeCourseAnalyzer()

    @pytest.fixture
    def sample_omics_data(self):
        """Create sample omics data."""
        return pd.DataFrame({
            "T0": [1.0, 2.0, 3.0, 4.0, 5.0],
            "T1": [1.5, 2.5, 3.5, 4.5, 5.5],
            "T2": [2.0, 3.0, 4.0, 5.0, 6.0],
            "T3": [2.5, 3.5, 4.5, 5.5, 6.5],
            "T4": [3.0, 4.0, 5.0, 6.0, 7.0]
        }, index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"])

    @pytest.fixture
    def sample_time_points(self):
        """Create sample time points."""
        return ["T0", "T1", "T2", "T3", "T4"]

    def test_init(self, time_course_analyzer):
        """Test TimeCourseAnalyzer initialization."""
        assert time_course_analyzer.logger is not None

    @pytest.mark.asyncio
    async def test_analyze_time_course_data_differential_expression(self, time_course_analyzer, sample_omics_data, sample_time_points):
        """Test time-course analysis with differential expression method."""
        result = await time_course_analyzer.analyze_time_course_data(
            omics_data=sample_omics_data,
            time_points=sample_time_points,
            analysis_method="differential_expression"
        )

        assert isinstance(result, dict)
        assert "differential_expression" in result
        assert len(result["differential_expression"]) > 0

    @pytest.mark.asyncio
    async def test_analyze_time_course_data_clustering(self, time_course_analyzer, sample_omics_data, sample_time_points):
        """Test time-course analysis with clustering method."""
        result = await time_course_analyzer.analyze_time_course_data(
            omics_data=sample_omics_data,
            time_points=sample_time_points,
            analysis_method="clustering"
        )

        assert isinstance(result, dict)
        assert "clustering" in result

    @pytest.mark.asyncio
    async def test_analyze_time_course_data_with_parameters(self, time_course_analyzer, sample_omics_data, sample_time_points):
        """Test time-course analysis with custom parameters."""
        parameters = {
            "threshold": 0.05,
            "method": "ttest"
        }

        result = await time_course_analyzer.analyze_time_course_data(
            omics_data=sample_omics_data,
            time_points=sample_time_points,
            analysis_method="differential_expression",
            parameters=parameters
        )

        assert isinstance(result, dict)
        assert "differential_expression" in result

    @pytest.mark.asyncio
    async def test_analyze_time_course_data_insufficient_time_points(self, time_course_analyzer, sample_omics_data):
        """Test time-course analysis with insufficient time points."""
        result = await time_course_analyzer.analyze_time_course_data(
            omics_data=sample_omics_data,
            time_points=["T0"],  # Only one time point
            analysis_method="differential_expression"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "Not enough time points" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_time_course_data_unknown_method(self, time_course_analyzer, sample_omics_data, sample_time_points):
        """Test time-course analysis with unknown method."""
        result = await time_course_analyzer.analyze_time_course_data(
            omics_data=sample_omics_data,
            time_points=sample_time_points,
            analysis_method="unknown_method"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "Unknown analysis method" in result["error"]

    def test_validate_omics_data(self, time_course_analyzer):
        """Test omics data validation."""
        # Valid omics data
        df = pd.DataFrame({"T0": [1, 2, 3], "T1": [2, 3, 4]})
        assert time_course_analyzer._validate_omics_data(df) is True
        
        # Invalid omics data
        assert time_course_analyzer._validate_omics_data(None) is False
        assert time_course_analyzer._validate_omics_data(pd.DataFrame()) is False

    def test_validate_time_points(self, time_course_analyzer):
        """Test time points validation."""
        # Valid time points
        assert time_course_analyzer._validate_time_points(["T0", "T1", "T2"]) is True
        
        # Invalid time points
        assert time_course_analyzer._validate_time_points([]) is False
        assert time_course_analyzer._validate_time_points(None) is False

    def test_validate_analysis_method(self, time_course_analyzer):
        """Test analysis method validation."""
        # Valid analysis methods
        assert time_course_analyzer._validate_analysis_method("differential_expression") is True
        assert time_course_analyzer._validate_analysis_method("clustering") is True
        
        # Invalid analysis method
        assert time_course_analyzer._validate_analysis_method("") is False
        assert time_course_analyzer._validate_analysis_method(None) is False

    def test_validate_parameters(self, time_course_analyzer):
        """Test parameters validation."""
        # Valid parameters
        assert time_course_analyzer._validate_parameters({"threshold": 0.05}) is True
        
        # Invalid parameters
        assert time_course_analyzer._validate_parameters(None) is False
