"""
Unit tests for the Data cache modules.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from pathwaylens_core.data.cache.cache_manager import CacheManager
from pathwaylens_core.data.cache.cache_strategies import (
    LRUCacheStrategy, FIFOCacheStrategy, LFUCacheStrategy
)
from pathwaylens_core.data.cache.cache_serializers import (
    JSONSerializer, PickleSerializer, ParquetSerializer
)


class TestCacheManager:
    """Test cases for the CacheManager class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create a CacheManager instance for testing."""
        return CacheManager(cache_dir=temp_cache_dir)

    def test_init(self, cache_manager):
        """Test CacheManager initialization."""
        assert cache_manager.logger is not None
        assert cache_manager.cache_dir is not None
        assert cache_manager.cache_strategy is not None
        assert cache_manager.serializer is not None

    def test_set_and_get(self, cache_manager):
        """Test setting and getting cache values."""
        key = "test_key"
        value = {"data": [1, 2, 3], "metadata": "test"}
        
        # Set value
        result = cache_manager.set(key, value)
        assert result is True
        
        # Get value
        retrieved_value = cache_manager.get(key)
        assert retrieved_value == value

    def test_set_and_get_dataframe(self, cache_manager):
        """Test setting and getting DataFrame cache values."""
        key = "test_df"
        value = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
        
        # Set value
        result = cache_manager.set(key, value)
        assert result is True
        
        # Get value
        retrieved_value = cache_manager.get(key)
        pd.testing.assert_frame_equal(retrieved_value, value)

    def test_get_nonexistent_key(self, cache_manager):
        """Test getting a nonexistent cache key."""
        result = cache_manager.get("nonexistent_key")
        assert result is None

    def test_delete(self, cache_manager):
        """Test deleting cache values."""
        key = "test_key"
        value = {"data": "test"}
        
        # Set value
        cache_manager.set(key, value)
        assert cache_manager.get(key) == value
        
        # Delete value
        result = cache_manager.delete(key)
        assert result is True
        
        # Verify deletion
        assert cache_manager.get(key) is None

    def test_clear(self, cache_manager):
        """Test clearing all cache values."""
        # Set multiple values
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")
        
        # Verify values exist
        assert cache_manager.get("key1") == "value1"
        assert cache_manager.get("key2") == "value2"
        assert cache_manager.get("key3") == "value3"
        
        # Clear cache
        result = cache_manager.clear()
        assert result is True
        
        # Verify all values are gone
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
        assert cache_manager.get("key3") is None

    def test_exists(self, cache_manager):
        """Test checking if cache key exists."""
        key = "test_key"
        value = {"data": "test"}
        
        # Key should not exist initially
        assert cache_manager.exists(key) is False
        
        # Set value
        cache_manager.set(key, value)
        
        # Key should exist now
        assert cache_manager.exists(key) is True

    def test_size(self, cache_manager):
        """Test getting cache size."""
        # Initially empty
        assert cache_manager.size() == 0
        
        # Add some values
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        # Should have 2 items
        assert cache_manager.size() == 2

    def test_keys(self, cache_manager):
        """Test getting cache keys."""
        # Initially empty
        assert len(cache_manager.keys()) == 0
        
        # Add some values
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        # Should have 2 keys
        keys = cache_manager.keys()
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    def test_values(self, cache_manager):
        """Test getting cache values."""
        # Initially empty
        assert len(cache_manager.values()) == 0
        
        # Add some values
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        # Should have 2 values
        values = cache_manager.values()
        assert len(values) == 2
        assert "value1" in values
        assert "value2" in values

    def test_items(self, cache_manager):
        """Test getting cache items."""
        # Initially empty
        assert len(cache_manager.items()) == 0
        
        # Add some values
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        # Should have 2 items
        items = cache_manager.items()
        assert len(items) == 2
        assert ("key1", "value1") in items
        assert ("key2", "value2") in items

    def test_validate_key(self, cache_manager):
        """Test key validation."""
        # Valid keys
        assert cache_manager._validate_key("valid_key") is True
        assert cache_manager._validate_key("key_with_underscores") is True
        assert cache_manager._validate_key("key-with-dashes") is True
        
        # Invalid keys
        assert cache_manager._validate_key("") is False
        assert cache_manager._validate_key(None) is False
        assert cache_manager._validate_key("key with spaces") is False

    def test_validate_value(self, cache_manager):
        """Test value validation."""
        # Valid values
        assert cache_manager._validate_value("string") is True
        assert cache_manager._validate_value(123) is True
        assert cache_manager._validate_value({"dict": "value"}) is True
        assert cache_manager._validate_value(pd.DataFrame({"col": [1, 2, 3]})) is True
        
        # Invalid values
        assert cache_manager._validate_value(None) is False

    def test_validate_cache_dir(self, cache_manager):
        """Test cache directory validation."""
        # Valid cache directory
        assert cache_manager._validate_cache_dir(cache_manager.cache_dir) is True
        
        # Invalid cache directory
        assert cache_manager._validate_cache_dir(None) is False
        assert cache_manager._validate_cache_dir(Path("/nonexistent/path")) is False


class TestLRUCacheStrategy:
    """Test cases for the LRUCacheStrategy class."""

    @pytest.fixture
    def lru_strategy(self):
        """Create an LRUCacheStrategy instance for testing."""
        return LRUCacheStrategy(max_size=3)

    def test_init(self, lru_strategy):
        """Test LRUCacheStrategy initialization."""
        assert lru_strategy.max_size == 3
        assert lru_strategy.cache is not None

    def test_put_and_get(self, lru_strategy):
        """Test putting and getting values."""
        # Put values
        lru_strategy.put("key1", "value1")
        lru_strategy.put("key2", "value2")
        lru_strategy.put("key3", "value3")
        
        # Get values
        assert lru_strategy.get("key1") == "value1"
        assert lru_strategy.get("key2") == "value2"
        assert lru_strategy.get("key3") == "value3"

    def test_lru_eviction(self, lru_strategy):
        """Test LRU eviction when cache is full."""
        # Fill cache
        lru_strategy.put("key1", "value1")
        lru_strategy.put("key2", "value2")
        lru_strategy.put("key3", "value3")
        
        # Access key1 to make it recently used
        lru_strategy.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        lru_strategy.put("key4", "value4")
        
        # key2 should be evicted
        assert lru_strategy.get("key2") is None
        # Other keys should still exist
        assert lru_strategy.get("key1") == "value1"
        assert lru_strategy.get("key3") == "value3"
        assert lru_strategy.get("key4") == "value4"

    def test_delete(self, lru_strategy):
        """Test deleting values."""
        lru_strategy.put("key1", "value1")
        lru_strategy.put("key2", "value2")
        
        # Delete key1
        result = lru_strategy.delete("key1")
        assert result is True
        
        # key1 should be gone
        assert lru_strategy.get("key1") is None
        # key2 should still exist
        assert lru_strategy.get("key2") == "value2"

    def test_clear(self, lru_strategy):
        """Test clearing cache."""
        lru_strategy.put("key1", "value1")
        lru_strategy.put("key2", "value2")
        
        # Clear cache
        lru_strategy.clear()
        
        # All keys should be gone
        assert lru_strategy.get("key1") is None
        assert lru_strategy.get("key2") is None

    def test_size(self, lru_strategy):
        """Test getting cache size."""
        # Initially empty
        assert lru_strategy.size() == 0
        
        # Add values
        lru_strategy.put("key1", "value1")
        lru_strategy.put("key2", "value2")
        
        # Should have 2 items
        assert lru_strategy.size() == 2

    def test_keys(self, lru_strategy):
        """Test getting cache keys."""
        # Initially empty
        assert len(lru_strategy.keys()) == 0
        
        # Add values
        lru_strategy.put("key1", "value1")
        lru_strategy.put("key2", "value2")
        
        # Should have 2 keys
        keys = lru_strategy.keys()
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys


class TestFIFOCacheStrategy:
    """Test cases for the FIFOCacheStrategy class."""

    @pytest.fixture
    def fifo_strategy(self):
        """Create a FIFOCacheStrategy instance for testing."""
        return FIFOCacheStrategy(max_size=3)

    def test_init(self, fifo_strategy):
        """Test FIFOCacheStrategy initialization."""
        assert fifo_strategy.max_size == 3
        assert fifo_strategy.cache is not None

    def test_put_and_get(self, fifo_strategy):
        """Test putting and getting values."""
        # Put values
        fifo_strategy.put("key1", "value1")
        fifo_strategy.put("key2", "value2")
        fifo_strategy.put("key3", "value3")
        
        # Get values
        assert fifo_strategy.get("key1") == "value1"
        assert fifo_strategy.get("key2") == "value2"
        assert fifo_strategy.get("key3") == "value3"

    def test_fifo_eviction(self, fifo_strategy):
        """Test FIFO eviction when cache is full."""
        # Fill cache
        fifo_strategy.put("key1", "value1")
        fifo_strategy.put("key2", "value2")
        fifo_strategy.put("key3", "value3")
        
        # Add new key, should evict key1 (first in)
        fifo_strategy.put("key4", "value4")
        
        # key1 should be evicted
        assert fifo_strategy.get("key1") is None
        # Other keys should still exist
        assert fifo_strategy.get("key2") == "value2"
        assert fifo_strategy.get("key3") == "value3"
        assert fifo_strategy.get("key4") == "value4"

    def test_delete(self, fifo_strategy):
        """Test deleting values."""
        fifo_strategy.put("key1", "value1")
        fifo_strategy.put("key2", "value2")
        
        # Delete key1
        result = fifo_strategy.delete("key1")
        assert result is True
        
        # key1 should be gone
        assert fifo_strategy.get("key1") is None
        # key2 should still exist
        assert fifo_strategy.get("key2") == "value2"

    def test_clear(self, fifo_strategy):
        """Test clearing cache."""
        fifo_strategy.put("key1", "value1")
        fifo_strategy.put("key2", "value2")
        
        # Clear cache
        fifo_strategy.clear()
        
        # All keys should be gone
        assert fifo_strategy.get("key1") is None
        assert fifo_strategy.get("key2") is None

    def test_size(self, fifo_strategy):
        """Test getting cache size."""
        # Initially empty
        assert fifo_strategy.size() == 0
        
        # Add values
        fifo_strategy.put("key1", "value1")
        fifo_strategy.put("key2", "value2")
        
        # Should have 2 items
        assert fifo_strategy.size() == 2


class TestLFUCacheStrategy:
    """Test cases for the LFUCacheStrategy class."""

    @pytest.fixture
    def lfu_strategy(self):
        """Create an LFUCacheStrategy instance for testing."""
        return LFUCacheStrategy(max_size=3)

    def test_init(self, lfu_strategy):
        """Test LFUCacheStrategy initialization."""
        assert lfu_strategy.max_size == 3
        assert lfu_strategy.cache is not None

    def test_put_and_get(self, lfu_strategy):
        """Test putting and getting values."""
        # Put values
        lfu_strategy.put("key1", "value1")
        lfu_strategy.put("key2", "value2")
        lfu_strategy.put("key3", "value3")
        
        # Get values
        assert lfu_strategy.get("key1") == "value1"
        assert lfu_strategy.get("key2") == "value2"
        assert lfu_strategy.get("key3") == "value3"

    def test_lfu_eviction(self, lfu_strategy):
        """Test LFU eviction when cache is full."""
        # Fill cache
        lfu_strategy.put("key1", "value1")
        lfu_strategy.put("key2", "value2")
        lfu_strategy.put("key3", "value3")
        
        # Access key1 and key2 multiple times to increase their frequency
        lfu_strategy.get("key1")
        lfu_strategy.get("key1")
        lfu_strategy.get("key2")
        lfu_strategy.get("key2")
        
        # Add new key, should evict key3 (least frequently used)
        lfu_strategy.put("key4", "value4")
        
        # key3 should be evicted
        assert lfu_strategy.get("key3") is None
        # Other keys should still exist
        assert lfu_strategy.get("key1") == "value1"
        assert lfu_strategy.get("key2") == "value2"
        assert lfu_strategy.get("key4") == "value4"

    def test_delete(self, lfu_strategy):
        """Test deleting values."""
        lfu_strategy.put("key1", "value1")
        lfu_strategy.put("key2", "value2")
        
        # Delete key1
        result = lfu_strategy.delete("key1")
        assert result is True
        
        # key1 should be gone
        assert lfu_strategy.get("key1") is None
        # key2 should still exist
        assert lfu_strategy.get("key2") == "value2"

    def test_clear(self, lfu_strategy):
        """Test clearing cache."""
        lfu_strategy.put("key1", "value1")
        lfu_strategy.put("key2", "value2")
        
        # Clear cache
        lfu_strategy.clear()
        
        # All keys should be gone
        assert lfu_strategy.get("key1") is None
        assert lfu_strategy.get("key2") is None

    def test_size(self, lfu_strategy):
        """Test getting cache size."""
        # Initially empty
        assert lfu_strategy.size() == 0
        
        # Add values
        lfu_strategy.put("key1", "value1")
        lfu_strategy.put("key2", "value2")
        
        # Should have 2 items
        assert lfu_strategy.size() == 2


class TestJSONSerializer:
    """Test cases for the JSONSerializer class."""

    @pytest.fixture
    def json_serializer(self):
        """Create a JSONSerializer instance for testing."""
        return JSONSerializer()

    def test_serialize_and_deserialize(self, json_serializer):
        """Test serializing and deserializing data."""
        data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        # Serialize
        serialized = json_serializer.serialize(data)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = json_serializer.deserialize(serialized)
        assert deserialized == data

    def test_serialize_and_deserialize_dataframe(self, json_serializer):
        """Test serializing and deserializing DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
        
        # Serialize
        serialized = json_serializer.serialize(df)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = json_serializer.deserialize(serialized)
        pd.testing.assert_frame_equal(deserialized, df)

    def test_serialize_invalid_data(self, json_serializer):
        """Test serializing invalid data."""
        # JSON cannot serialize functions
        def test_func():
            pass
        
        with pytest.raises(TypeError):
            json_serializer.serialize(test_func)

    def test_deserialize_invalid_data(self, json_serializer):
        """Test deserializing invalid data."""
        # Invalid JSON
        invalid_json = b"invalid json content"
        
        with pytest.raises(ValueError):
            json_serializer.deserialize(invalid_json)


class TestPickleSerializer:
    """Test cases for the PickleSerializer class."""

    @pytest.fixture
    def pickle_serializer(self):
        """Create a PickleSerializer instance for testing."""
        return PickleSerializer()

    def test_serialize_and_deserialize(self, pickle_serializer):
        """Test serializing and deserializing data."""
        data = {"key": "value", "number": 123, "list": [1, 2, 3]}
        
        # Serialize
        serialized = pickle_serializer.serialize(data)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = pickle_serializer.deserialize(serialized)
        assert deserialized == data

    def test_serialize_and_deserialize_dataframe(self, pickle_serializer):
        """Test serializing and deserializing DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
        
        # Serialize
        serialized = pickle_serializer.serialize(df)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = pickle_serializer.deserialize(serialized)
        pd.testing.assert_frame_equal(deserialized, df)

    def test_serialize_and_deserialize_function(self, pickle_serializer):
        """Test serializing and deserializing function."""
        def test_func(x):
            return x * 2
        
        # Serialize
        serialized = pickle_serializer.serialize(test_func)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = pickle_serializer.deserialize(serialized)
        assert deserialized(5) == 10

    def test_deserialize_invalid_data(self, pickle_serializer):
        """Test deserializing invalid data."""
        # Invalid pickle data
        invalid_pickle = b"invalid pickle content"
        
        with pytest.raises(Exception):
            pickle_serializer.deserialize(invalid_pickle)


class TestParquetSerializer:
    """Test cases for the ParquetSerializer class."""

    @pytest.fixture
    def parquet_serializer(self):
        """Create a ParquetSerializer instance for testing."""
        return ParquetSerializer()

    def test_serialize_and_deserialize_dataframe(self, parquet_serializer):
        """Test serializing and deserializing DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["A", "B", "C"]})
        
        # Serialize
        serialized = parquet_serializer.serialize(df)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = parquet_serializer.deserialize(serialized)
        pd.testing.assert_frame_equal(deserialized, df)

    def test_serialize_invalid_data(self, parquet_serializer):
        """Test serializing invalid data."""
        # Parquet can only serialize DataFrames
        invalid_data = {"key": "value"}
        
        with pytest.raises(TypeError):
            parquet_serializer.serialize(invalid_data)

    def test_deserialize_invalid_data(self, parquet_serializer):
        """Test deserializing invalid data."""
        # Invalid parquet data
        invalid_parquet = b"invalid parquet content"
        
        with pytest.raises(Exception):
            parquet_serializer.deserialize(invalid_parquet)