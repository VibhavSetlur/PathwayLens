"""
Unit tests for the Data utils modules.
"""

import pytest
import pandas as pd
import numpy as np
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from pathwaylens_core.data.utils.database_utils import (
    load_database_config, save_database_config, validate_database_config,
    get_database_connection, close_database_connection
)
from pathwaylens_core.data.utils.file_utils import (
    read_file_content, write_file_content, validate_file_path,
    get_file_extension, get_file_size, create_directory
)
from pathwaylens_core.data.utils.validation_utils import (
    validate_gene_list, validate_expression_data, validate_pathway_data,
    validate_metadata, validate_data_format
)


class TestDatabaseUtils:
    """Test cases for the database utilities."""

    def test_load_database_config(self, tmp_path):
        """Test loading database configuration."""
        config_file = tmp_path / "test_config.yml"
        config_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "user": "test_user",
                "password": "test_password"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        result = load_database_config(str(config_file))
        
        assert isinstance(result, dict)
        assert "database" in result
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == 5432

    def test_load_database_config_invalid_file(self, tmp_path):
        """Test loading database configuration from invalid file."""
        config_file = tmp_path / "invalid_config.yml"
        config_file.write_text("invalid: yaml: content: [unclosed")
        
        result = load_database_config(str(config_file))
        
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_save_database_config(self, tmp_path):
        """Test saving database configuration."""
        config_file = tmp_path / "test_config.yml"
        config_data = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db"
            }
        }
        
        result = save_database_config(config_data, str(config_file))
        
        assert result is True
        assert config_file.exists()
        
        # Verify content
        with open(config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        assert saved_config == config_data

    def test_validate_database_config(self):
        """Test database configuration validation."""
        # Valid config
        valid_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "user": "test_user",
                "password": "test_password"
            }
        }
        assert validate_database_config(valid_config) is True
        
        # Invalid config - missing required fields
        invalid_config = {
            "database": {
                "host": "localhost"
            }
        }
        assert validate_database_config(invalid_config) is False
        
        # Invalid config - empty
        assert validate_database_config({}) is False
        assert validate_database_config(None) is False

    def test_get_database_connection(self):
        """Test getting database connection."""
        # This would require actual database setup, so we'll test the function exists
        # and returns a connection object or raises an appropriate error
        try:
            connection = get_database_connection({
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "user": "test_user",
                "password": "test_password"
            })
            # If connection succeeds, it should be a connection object
            assert connection is not None
        except Exception as e:
            # If connection fails, it should be a database-related error
            assert "database" in str(e).lower() or "connection" in str(e).lower()

    def test_close_database_connection(self):
        """Test closing database connection."""
        # Mock connection object
        mock_connection = Mock()
        mock_connection.close = Mock()
        
        result = close_database_connection(mock_connection)
        
        assert result is True
        mock_connection.close.assert_called_once()


class TestFileUtils:
    """Test cases for the file utilities."""

    def test_read_file_content_json(self, tmp_path):
        """Test reading JSON file content."""
        json_file = tmp_path / "test.json"
        json_data = {"key": "value", "number": 123}
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f)

        result = read_file_content(json_file)
        
        assert result == json_data

    def test_read_file_content_yaml(self, tmp_path):
        """Test reading YAML file content."""
        yaml_file = tmp_path / "test.yml"
        yaml_data = {"key": "value", "number": 123}
        
        with open(yaml_file, 'w') as f:
            yaml.dump(yaml_data, f)

        result = read_file_content(yaml_file)
        
        assert result == yaml_data

    def test_read_file_content_csv(self, tmp_path):
        """Test reading CSV file content."""
        csv_file = tmp_path / "test.csv"
        csv_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
        csv_data.to_csv(csv_file, index=False)

        result = read_file_content(csv_file)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "col1" in result.columns
        assert "col2" in result.columns

    def test_read_file_content_text(self, tmp_path):
        """Test reading text file content."""
        text_file = tmp_path / "test.txt"
        text_content = "This is a test file content."
        text_file.write_text(text_content)

        result = read_file_content(text_file)
        
        assert result == text_content

    def test_read_file_content_invalid_file(self, tmp_path):
        """Test reading content from invalid file."""
        invalid_file = tmp_path / "nonexistent.txt"
        
        result = read_file_content(invalid_file)
        
        assert result is None

    def test_write_file_content_json(self, tmp_path):
        """Test writing JSON file content."""
        json_file = tmp_path / "test.json"
        json_data = {"key": "value", "number": 123}
        
        result = write_file_content(json_data, json_file)
        
        assert result is True
        assert json_file.exists()
        
        # Verify content
        with open(json_file, 'r') as f:
            saved_data = json.load(f)
        assert saved_data == json_data

    def test_write_file_content_yaml(self, tmp_path):
        """Test writing YAML file content."""
        yaml_file = tmp_path / "test.yml"
        yaml_data = {"key": "value", "number": 123}
        
        result = write_file_content(yaml_data, yaml_file)
        
        assert result is True
        assert yaml_file.exists()
        
        # Verify content
        with open(yaml_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        assert saved_data == yaml_data

    def test_write_file_content_csv(self, tmp_path):
        """Test writing CSV file content."""
        csv_file = tmp_path / "test.csv"
        csv_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
        
        result = write_file_content(csv_data, csv_file)
        
        assert result is True
        assert csv_file.exists()
        
        # Verify content
        saved_data = pd.read_csv(csv_file)
        pd.testing.assert_frame_equal(csv_data, saved_data)

    def test_write_file_content_text(self, tmp_path):
        """Test writing text file content."""
        text_file = tmp_path / "test.txt"
        text_content = "This is a test file content."
        
        result = write_file_content(text_content, text_file)
        
        assert result is True
        assert text_file.exists()
        assert text_file.read_text() == text_content

    def test_validate_file_path(self, tmp_path):
        """Test file path validation."""
        # Valid file path
        valid_file = tmp_path / "test.txt"
        valid_file.write_text("test content")
        assert validate_file_path(valid_file) is True
        
        # Invalid file path
        invalid_file = tmp_path / "nonexistent.txt"
        assert validate_file_path(invalid_file) is False
        
        # None path
        assert validate_file_path(None) is False

    def test_get_file_extension(self, tmp_path):
        """Test getting file extension."""
        # Test various file extensions
        assert get_file_extension("test.txt") == ".txt"
        assert get_file_extension("test.json") == ".json"
        assert get_file_extension("test.yml") == ".yml"
        assert get_file_extension("test.csv") == ".csv"
        assert get_file_extension("test") == ""
        assert get_file_extension("") == ""

    def test_get_file_size(self, tmp_path):
        """Test getting file size."""
        test_file = tmp_path / "test.txt"
        test_content = "This is a test file content."
        test_file.write_text(test_content)
        
        size = get_file_size(test_file)
        
        assert size == len(test_content.encode('utf-8'))

    def test_create_directory(self, tmp_path):
        """Test creating directory."""
        new_dir = tmp_path / "new_directory"
        
        result = create_directory(new_dir)
        
        assert result is True
        assert new_dir.exists()
        assert new_dir.is_dir()


class TestValidationUtils:
    """Test cases for the validation utilities."""

    def test_validate_gene_list(self):
        """Test gene list validation."""
        # Valid gene list
        valid_genes = ["GENE1", "GENE2", "GENE3"]
        assert validate_gene_list(valid_genes) is True
        
        # Invalid gene list
        assert validate_gene_list([]) is False
        assert validate_gene_list(None) is False
        assert validate_gene_list(["", "GENE2"]) is False

    def test_validate_expression_data(self):
        """Test expression data validation."""
        # Valid expression data
        valid_data = pd.DataFrame({
            "GENE1": [1.0, 2.0, 3.0],
            "GENE2": [2.0, 4.0, 6.0]
        })
        assert validate_expression_data(valid_data) is True
        
        # Invalid expression data
        assert validate_expression_data(pd.DataFrame()) is False
        assert validate_expression_data(None) is False
        
        # Data with non-numeric values
        invalid_data = pd.DataFrame({
            "GENE1": ["A", "B", "C"],
            "GENE2": [1, 2, 3]
        })
        assert validate_expression_data(invalid_data) is False

    def test_validate_pathway_data(self):
        """Test pathway data validation."""
        # Valid pathway data
        valid_pathways = {
            "PATH:00010": {"name": "Glycolysis", "genes": ["GENE1", "GENE2"]},
            "PATH:00020": {"name": "TCA Cycle", "genes": ["GENE3", "GENE4"]}
        }
        assert validate_pathway_data(valid_pathways) is True
        
        # Invalid pathway data
        assert validate_pathway_data({}) is False
        assert validate_pathway_data(None) is False
        
        # Data with missing required fields
        invalid_pathways = {
            "PATH:00010": {"name": "Glycolysis"},  # Missing genes
            "PATH:00020": {"genes": ["GENE3", "GENE4"]}  # Missing name
        }
        assert validate_pathway_data(invalid_pathways) is False

    def test_validate_metadata(self):
        """Test metadata validation."""
        # Valid metadata
        valid_metadata = {
            "species": "human",
            "tissue": "liver",
            "condition": "control",
            "replicates": 3
        }
        assert validate_metadata(valid_metadata) is True
        
        # Invalid metadata
        assert validate_metadata({}) is False
        assert validate_metadata(None) is False
        
        # Metadata with missing required fields
        invalid_metadata = {
            "species": "human",
            "tissue": "liver"
            # Missing condition and replicates
        }
        assert validate_metadata(invalid_metadata) is False

    def test_validate_data_format(self):
        """Test data format validation."""
        # Valid data formats
        assert validate_data_format("gene_list") is True
        assert validate_data_format("expression_matrix") is True
        assert validate_data_format("pathway_data") is True
        assert validate_data_format("metadata") is True
        
        # Invalid data formats
        assert validate_data_format("") is False
        assert validate_data_format(None) is False
        assert validate_data_format("invalid_format") is False