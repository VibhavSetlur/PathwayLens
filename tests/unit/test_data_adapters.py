"""
Unit tests for the Data adapters.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.data.adapters.kegg_adapter import KEGGAdapter
from pathwaylens_core.data.adapters.reactome_adapter import ReactomeAdapter
from pathwaylens_core.data.adapters.go_adapter import GOAdapter
from pathwaylens_core.data.adapters.wikipathways_adapter import WikiPathwaysAdapter
from pathwaylens_core.data.adapters.msigdb_adapter import MSigDBAdapter
from pathwaylens_core.data.adapters.pathbank_adapter import PathBankAdapter
from pathwaylens_core.data.adapters.netpath_adapter import NetPathAdapter
from pathwaylens_core.data.adapters.panther_adapter import PantherAdapter
from pathwaylens_core.data.adapters.hallmark_adapter import HallmarkAdapter
from pathwaylens_core.data.adapters.custom_adapter import CustomAdapter


class TestKEGGAdapter:
    """Test cases for the KEGGAdapter class."""

    @pytest.fixture
    def kegg_adapter(self):
        """Create a KEGGAdapter instance for testing."""
        return KEGGAdapter()

    def test_init(self, kegg_adapter):
        """Test KEGGAdapter initialization."""
        assert kegg_adapter.logger is not None
        assert kegg_adapter.database_name == "KEGG"
        assert kegg_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, kegg_adapter):
        """Test getting pathway data from KEGG."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "hsa00010", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "hsa00020", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await kegg_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "hsa00010" in result
            assert result["hsa00010"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, kegg_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await kegg_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, kegg_adapter):
        """Test species validation."""
        # Valid species
        assert kegg_adapter._validate_species("human") is True
        assert kegg_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert kegg_adapter._validate_species("") is False
        assert kegg_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, kegg_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "hsa00010": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert kegg_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert kegg_adapter._validate_pathway_data({}) is False
        assert kegg_adapter._validate_pathway_data(None) is False


class TestReactomeAdapter:
    """Test cases for the ReactomeAdapter class."""

    @pytest.fixture
    def reactome_adapter(self):
        """Create a ReactomeAdapter instance for testing."""
        return ReactomeAdapter()

    def test_init(self, reactome_adapter):
        """Test ReactomeAdapter initialization."""
        assert reactome_adapter.logger is not None
        assert reactome_adapter.database_name == "Reactome"
        assert reactome_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, reactome_adapter):
        """Test getting pathway data from Reactome."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "R-HSA-1430728", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "R-HSA-71403", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await reactome_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "R-HSA-1430728" in result
            assert result["R-HSA-1430728"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, reactome_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await reactome_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, reactome_adapter):
        """Test species validation."""
        # Valid species
        assert reactome_adapter._validate_species("human") is True
        assert reactome_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert reactome_adapter._validate_species("") is False
        assert reactome_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, reactome_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "R-HSA-1430728": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert reactome_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert reactome_adapter._validate_pathway_data({}) is False
        assert reactome_adapter._validate_pathway_data(None) is False


class TestGOAdapter:
    """Test cases for the GOAdapter class."""

    @pytest.fixture
    def go_adapter(self):
        """Create a GOAdapter instance for testing."""
        return GOAdapter()

    def test_init(self, go_adapter):
        """Test GOAdapter initialization."""
        assert go_adapter.logger is not None
        assert go_adapter.database_name == "GO"
        assert go_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, go_adapter):
        """Test getting pathway data from GO."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "GO:0006096", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "GO:0006099", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await go_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "GO:0006096" in result
            assert result["GO:0006096"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, go_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await go_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, go_adapter):
        """Test species validation."""
        # Valid species
        assert go_adapter._validate_species("human") is True
        assert go_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert go_adapter._validate_species("") is False
        assert go_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, go_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "GO:0006096": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert go_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert go_adapter._validate_pathway_data({}) is False
        assert go_adapter._validate_pathway_data(None) is False


class TestWikiPathwaysAdapter:
    """Test cases for the WikiPathwaysAdapter class."""

    @pytest.fixture
    def wikipathways_adapter(self):
        """Create a WikiPathwaysAdapter instance for testing."""
        return WikiPathwaysAdapter()

    def test_init(self, wikipathways_adapter):
        """Test WikiPathwaysAdapter initialization."""
        assert wikipathways_adapter.logger is not None
        assert wikipathways_adapter.database_name == "WikiPathways"
        assert wikipathways_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, wikipathways_adapter):
        """Test getting pathway data from WikiPathways."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "WP1", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "WP2", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await wikipathways_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "WP1" in result
            assert result["WP1"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, wikipathways_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await wikipathways_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, wikipathways_adapter):
        """Test species validation."""
        # Valid species
        assert wikipathways_adapter._validate_species("human") is True
        assert wikipathways_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert wikipathways_adapter._validate_species("") is False
        assert wikipathways_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, wikipathways_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "WP1": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert wikipathways_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert wikipathways_adapter._validate_pathway_data({}) is False
        assert wikipathways_adapter._validate_pathway_data(None) is False


class TestMSigDBAdapter:
    """Test cases for the MSigDBAdapter class."""

    @pytest.fixture
    def msigdb_adapter(self):
        """Create a MSigDBAdapter instance for testing."""
        return MSigDBAdapter()

    def test_init(self, msigdb_adapter):
        """Test MSigDBAdapter initialization."""
        assert msigdb_adapter.logger is not None
        assert msigdb_adapter.database_name == "MSigDB"
        assert msigdb_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, msigdb_adapter):
        """Test getting pathway data from MSigDB."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "M1", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "M2", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await msigdb_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "M1" in result
            assert result["M1"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, msigdb_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await msigdb_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, msigdb_adapter):
        """Test species validation."""
        # Valid species
        assert msigdb_adapter._validate_species("human") is True
        assert msigdb_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert msigdb_adapter._validate_species("") is False
        assert msigdb_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, msigdb_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "M1": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert msigdb_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert msigdb_adapter._validate_pathway_data({}) is False
        assert msigdb_adapter._validate_pathway_data(None) is False


class TestPathBankAdapter:
    """Test cases for the PathBankAdapter class."""

    @pytest.fixture
    def pathbank_adapter(self):
        """Create a PathBankAdapter instance for testing."""
        return PathBankAdapter()

    def test_init(self, pathbank_adapter):
        """Test PathBankAdapter initialization."""
        assert pathbank_adapter.logger is not None
        assert pathbank_adapter.database_name == "PathBank"
        assert pathbank_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, pathbank_adapter):
        """Test getting pathway data from PathBank."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "PB1", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "PB2", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await pathbank_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "PB1" in result
            assert result["PB1"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, pathbank_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await pathbank_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, pathbank_adapter):
        """Test species validation."""
        # Valid species
        assert pathbank_adapter._validate_species("human") is True
        assert pathbank_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert pathbank_adapter._validate_species("") is False
        assert pathbank_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, pathbank_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "PB1": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert pathbank_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert pathbank_adapter._validate_pathway_data({}) is False
        assert pathbank_adapter._validate_pathway_data(None) is False


class TestNetPathAdapter:
    """Test cases for the NetPathAdapter class."""

    @pytest.fixture
    def netpath_adapter(self):
        """Create a NetPathAdapter instance for testing."""
        return NetPathAdapter()

    def test_init(self, netpath_adapter):
        """Test NetPathAdapter initialization."""
        assert netpath_adapter.logger is not None
        assert netpath_adapter.database_name == "NetPath"
        assert netpath_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, netpath_adapter):
        """Test getting pathway data from NetPath."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "NP1", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "NP2", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await netpath_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "NP1" in result
            assert result["NP1"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, netpath_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await netpath_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, netpath_adapter):
        """Test species validation."""
        # Valid species
        assert netpath_adapter._validate_species("human") is True
        assert netpath_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert netpath_adapter._validate_species("") is False
        assert netpath_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, netpath_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "NP1": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert netpath_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert netpath_adapter._validate_pathway_data({}) is False
        assert netpath_adapter._validate_pathway_data(None) is False


class TestPantherAdapter:
    """Test cases for the PantherAdapter class."""

    @pytest.fixture
    def panther_adapter(self):
        """Create a PantherAdapter instance for testing."""
        return PantherAdapter()

    def test_init(self, panther_adapter):
        """Test PantherAdapter initialization."""
        assert panther_adapter.logger is not None
        assert panther_adapter.database_name == "Panther"
        assert panther_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, panther_adapter):
        """Test getting pathway data from Panther."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "P1", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "P2", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await panther_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "P1" in result
            assert result["P1"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, panther_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await panther_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, panther_adapter):
        """Test species validation."""
        # Valid species
        assert panther_adapter._validate_species("human") is True
        assert panther_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert panther_adapter._validate_species("") is False
        assert panther_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, panther_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "P1": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert panther_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert panther_adapter._validate_pathway_data({}) is False
        assert panther_adapter._validate_pathway_data(None) is False


class TestHallmarkAdapter:
    """Test cases for the HallmarkAdapter class."""

    @pytest.fixture
    def hallmark_adapter(self):
        """Create a HallmarkAdapter instance for testing."""
        return HallmarkAdapter()

    def test_init(self, hallmark_adapter):
        """Test HallmarkAdapter initialization."""
        assert hallmark_adapter.logger is not None
        assert hallmark_adapter.database_name == "Hallmark"
        assert hallmark_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, hallmark_adapter):
        """Test getting pathway data from Hallmark."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "H1", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "H2", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await hallmark_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "H1" in result
            assert result["H1"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, hallmark_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await hallmark_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, hallmark_adapter):
        """Test species validation."""
        # Valid species
        assert hallmark_adapter._validate_species("human") is True
        assert hallmark_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert hallmark_adapter._validate_species("") is False
        assert hallmark_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, hallmark_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "H1": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert hallmark_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert hallmark_adapter._validate_pathway_data({}) is False
        assert hallmark_adapter._validate_pathway_data(None) is False


class TestCustomAdapter:
    """Test cases for the CustomAdapter class."""

    @pytest.fixture
    def custom_adapter(self):
        """Create a CustomAdapter instance for testing."""
        return CustomAdapter()

    def test_init(self, custom_adapter):
        """Test CustomAdapter initialization."""
        assert custom_adapter.logger is not None
        assert custom_adapter.database_name == "Custom"
        assert custom_adapter.supported_species is not None

    @pytest.mark.asyncio
    async def test_get_pathway_data(self, custom_adapter):
        """Test getting pathway data from Custom adapter."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "pathways": [
                    {"id": "C1", "name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
                    {"id": "C2", "name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
                ]
            }
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = await custom_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert len(result) > 0
            assert "C1" in result
            assert result["C1"]["name"] == "Glycolysis"

    @pytest.mark.asyncio
    async def test_get_pathway_data_api_error(self, custom_adapter):
        """Test getting pathway data with API error."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("API Error")

            result = await custom_adapter.get_pathway_data(species="human")

            assert isinstance(result, dict)
            assert "error" in result
            assert "API Error" in result["error"]

    def test_validate_species(self, custom_adapter):
        """Test species validation."""
        # Valid species
        assert custom_adapter._validate_species("human") is True
        assert custom_adapter._validate_species("mouse") is True
        
        # Invalid species
        assert custom_adapter._validate_species("") is False
        assert custom_adapter._validate_species(None) is False

    def test_validate_pathway_data(self, custom_adapter):
        """Test pathway data validation."""
        # Valid pathway data
        pathway_data = {
            "C1": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]}
        }
        assert custom_adapter._validate_pathway_data(pathway_data) is True
        
        # Invalid pathway data
        assert custom_adapter._validate_pathway_data({}) is False
        assert custom_adapter._validate_pathway_data(None) is False