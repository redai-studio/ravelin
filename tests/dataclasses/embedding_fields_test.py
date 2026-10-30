import unittest
from unittest.mock import patch, MagicMock

from ravelin.dataclasses.asset_reference import AssetReference
from ravelin.dataclasses.embedding_fields import (
    EmbeddingField,
    VectorType,
)
from ravelin.embedding_functions._base import RegisteredEmbeddingFunction


class TestEmbeddingField(unittest.TestCase):
    def setUp(self):
        self.asset_reference = AssetReference(name="test_asset", source="local")
        self.mock_embedding_function = MagicMock(spec=RegisteredEmbeddingFunction)
        self.mock_embedding_function.to_dict.return_value = {"type": "test_function"}

        # Initialize with the setter property
        self.embedding_field = EmbeddingField(search_kwargs={"arg2": "value2"})
        self.embedding_field.embedding_function = self.mock_embedding_function

    def test_initialization(self):
        """Test that EmbeddingField initializes correctly with various parameters."""
        self.assertEqual(self.mock_embedding_function, self.embedding_field.embedding_function)
        self.assertEqual({"arg2": "value2"}, self.embedding_field.search_kwargs)
        # Check default vector type
        self.assertEqual(VectorType.DENSE, self.embedding_field.vector_type)

        # Test initialization with None embedding_function
        field_no_func = EmbeddingField(search_kwargs={"arg2": "value2"})
        self.assertIsNone(field_no_func.embedding_function)

    def test_embedding_function_property(self):
        """Test the embedding_function property getter and setter."""
        # Test setting to None
        self.embedding_field.embedding_function = None
        self.assertIsNone(self.embedding_field.embedding_function)
        self.assertIsNone(self.embedding_field._embedding_function_data)

        # Test setting a new value
        new_mock = MagicMock(spec=RegisteredEmbeddingFunction)
        new_mock.to_dict.return_value = {"type": "new_function"}
        self.embedding_field.embedding_function = new_mock
        self.assertEqual(new_mock, self.embedding_field.embedding_function)
        self.assertEqual({"type": "new_function"}, self.embedding_field._embedding_function_data)

    def test_lazy_loading(self):
        """Test lazy loading of embedding function from serialized data."""
        field = EmbeddingField()
        # Directly set the private attribute
        field._embedding_function_data = {"type": "lazy_loaded"}

        # The function shouldn't be loaded yet
        self.assertIsNone(field._loaded_embedding_function)

        with patch("ravelin.embedding_functions.factory.EmbeddingFunctionFactory.create_from_dict") as mock_create:
            mock_ef = MagicMock(spec=RegisteredEmbeddingFunction)
            mock_create.return_value = mock_ef

            # This should trigger lazy loading
            result = field.embedding_function

            mock_create.assert_called_once_with({"type": "lazy_loaded"})
            self.assertEqual(mock_ef, result)
            self.assertEqual(mock_ef, field._loaded_embedding_function)

    def test_legacy_compatibility(self):
        """Test compatibility with legacy fields using the metaclass."""
        with patch(
            "ravelin.embedding_functions.factory.EmbeddingFunctionFactory.create_from_dict"
        ) as mock_create_from_dict:
            mock_embedding_function = MagicMock(spec=RegisteredEmbeddingFunction)
            mock_create_from_dict.return_value = mock_embedding_function

            # Instead of directly constructing with legacy parameters,
            # we'll test if the _BackwardCompatible metaclass works through from_dict
            legacy_data = {
                "from_asset": self.asset_reference.to_dict(),
                "embedder_kwargs": {"arg1": "value1"},
                "search_kwargs": {"arg2": "value2"},
            }

            # Create via from_dict which will use the metaclass logic
            field = EmbeddingField.from_dict(legacy_data)

            # Access embedding_function to trigger lazy loading
            self.assertEqual(mock_embedding_function, field.embedding_function)

            # Verify create_from_dict was called with the right parameters
            expected_data = {
                "embedding_function_type": "model_service",
                "model_service": self.asset_reference.to_dict(),
                "timeout": None,
                "model": None,
                "embedder_kwargs": {"arg1": "value1"},
            }
            mock_create_from_dict.assert_called_once()
            # Check keys are the same (order might be different)
            self.assertEqual(set(expected_data.keys()), set(mock_create_from_dict.call_args[0][0].keys()))

    def test_to_dict(self):
        """Test converting EmbeddingField to dictionary."""
        expected_dict = {
            "embedding_function": {"type": "test_function"},
            "search_kwargs": {"arg2": "value2"},
            "vector_type": "dense",  # Default value should be included
        }
        self.assertEqual(expected_dict, self.embedding_field.to_dict())

        # Test to_dict with None embedding_function
        field_no_func = EmbeddingField(search_kwargs={"arg2": "value2"})
        self.assertNotIn("embedding_function", field_no_func.to_dict())

    @patch("ravelin.embedding_functions.factory.EmbeddingFunctionFactory.create_from_dict")
    def test_from_dict(self, mock_create_from_dict):
        """Test creating EmbeddingField from dictionary."""
        mock_embedding_function = MagicMock(spec=RegisteredEmbeddingFunction)
        mock_create_from_dict.return_value = mock_embedding_function

        data = {
            "embedding_function": {"type": "test_function"},
            "search_kwargs": {"arg2": "value2"},
        }

        embedding_field = EmbeddingField.from_dict(data)

        # Initially, the function shouldn't be loaded
        self.assertIsNone(embedding_field._loaded_embedding_function)

        # Accessing the property should trigger loading
        self.assertEqual(mock_embedding_function, embedding_field.embedding_function)
        mock_create_from_dict.assert_called_once_with({"type": "test_function"})
        self.assertEqual({"arg2": "value2"}, embedding_field.search_kwargs)

        # Test from_dict with no embedding_function
        data_no_func = {
            "search_kwargs": {"arg2": "value2"},
        }

        mock_create_from_dict.reset_mock()
        field_no_func = EmbeddingField.from_dict(data_no_func)
        # We don't access the property, so create_from_dict shouldn't be called
        mock_create_from_dict.assert_not_called()
        # Explicitly access to confirm it's None
        self.assertIsNone(field_no_func.embedding_function)
        self.assertEqual({"arg2": "value2"}, field_no_func.search_kwargs)

    @patch("ravelin.dataclasses.asset_reference.AssetReference.from_dict")
    def test_from_dict_with_legacy_fields(self, mock_from_dict):
        """Test creating EmbeddingField from dictionary with legacy fields."""
        mock_asset_ref = MagicMock(spec=AssetReference)
        mock_asset_ref.to_dict.return_value = {"name": "test_asset", "source": "local"}
        mock_from_dict.return_value = mock_asset_ref

        data = {
            "from_asset": {"name": "test_asset", "source": "local"},
            "embedder_kwargs": {"arg1": "value1"},
            "search_kwargs": {"arg2": "value2"},
        }

        with patch("ravelin.embedding_functions.factory.EmbeddingFunctionFactory.create_from_dict") as mock_create:
            mock_ef = MagicMock(spec=RegisteredEmbeddingFunction)
            mock_create.return_value = mock_ef

            embedding_field = EmbeddingField.from_dict(data)

            # Verify data was correctly set up
            expected_data = {
                "embedding_function_type": "model_service",
                "model_service": {"name": "test_asset", "source": "local"},
                "timeout": None,
                "model": None,
                "embedder_kwargs": {"arg1": "value1"},
            }

            # Access property to trigger loading
            self.assertEqual(mock_ef, embedding_field.embedding_function)
            mock_create.assert_called_once()
            # Check the keys are the same (order might differ)
            self.assertEqual(set(expected_data.keys()), set(mock_create.call_args[0][0].keys()))
            self.assertEqual({"arg2": "value2"}, embedding_field.search_kwargs)

    def test_vector_type_enum(self):
        """Test the VectorType enum values."""
        self.assertEqual("dense", VectorType.DENSE.value)
        self.assertEqual("sparse", VectorType.SPARSE.value)

        # Test creating with explicit vector type
        sparse_field = EmbeddingField(vector_type=VectorType.SPARSE)
        sparse_field.embedding_function = self.mock_embedding_function
        self.assertEqual(VectorType.SPARSE, sparse_field.vector_type)

    def test_from_dict_with_vector_type(self):
        """Test creating an EmbeddingField from a dict with vector_type specified."""
        data = {
            "embedding_function": {"type": "test_function"},
            "vector_type": "sparse",
        }

        with patch("ravelin.embedding_functions.factory.EmbeddingFunctionFactory.create_from_dict") as mock_create:
            mock_embedding_function = MagicMock(spec=RegisteredEmbeddingFunction)
            mock_create.return_value = mock_embedding_function

            field = EmbeddingField.from_dict(data)
            self.assertEqual(VectorType.SPARSE, field.vector_type)
            self.assertEqual(mock_embedding_function, field.embedding_function)
