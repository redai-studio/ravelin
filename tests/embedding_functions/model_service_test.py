"""Unit tests for ModelServiceEmbeddingFunction."""

import unittest
from unittest.mock import MagicMock, patch

from ravelin.assets.model_services import EmbeddingModelService
from ravelin.dataclasses import AssetReference
from ravelin.embedding_functions import ModelServiceEmbeddingFunction


class ModelServiceEmbeddingFunctionTest(unittest.TestCase):
    """Test suite for ModelServiceEmbeddingFunction."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a mock embedding model service
        self.mock_model_service = MagicMock(spec=EmbeddingModelService)
        self.mock_model_service.infer.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        # Create a mock reference for serialization tests
        self.mock_model_service.reference = MagicMock(spec=AssetReference)
        self.mock_model_service.reference.to_dict.return_value = {"id": "test-model-service"}

        # Create the embedding function
        self.embedding_fn: ModelServiceEmbeddingFunction = ModelServiceEmbeddingFunction(
            model_service=self.mock_model_service, timeout=10.0, model="test-model", embedder_kwargs={"normalize": True}
        )

    def test_initialization(self) -> None:
        """Test proper initialization of the embedding function."""
        self.assertEqual(self.mock_model_service, self.embedding_fn.model_service)
        self.assertEqual(10.0, self.embedding_fn.timeout)
        self.assertEqual("test-model", self.embedding_fn.model)
        self.assertEqual({"normalize": True}, self.embedding_fn.embedder_kwargs)

    def test_call(self) -> None:
        """Test the __call__ method to generate embeddings."""
        inputs = ["test text 1", "test text 2"]
        embeddings = self.embedding_fn(inputs)

        # Verify model service was called with correct parameters
        self.mock_model_service.infer.assert_called_once_with(
            inputs=inputs, timeout=10.0, model="test-model", inference_kwargs={"normalize": True}
        )

        # Verify the embeddings were returned correctly
        self.assertEqual([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], embeddings)

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        serialized = self.embedding_fn.to_dict()

        # Check the serialized output has all required fields
        self.assertEqual("model-service", serialized["embedding_function_type"])
        self.assertEqual({"id": "test-model-service"}, serialized["model_service"])
        self.assertEqual(10.0, serialized["timeout"])
        self.assertEqual("test-model", serialized["model"])
        self.assertEqual({"normalize": True}, serialized["embedder_kwargs"])

    @patch("ravelin.embedding_functions.model_service.AssetReference.from_dict")
    @patch("ravelin.embedding_functions.model_service.ModelServiceFactory.load_from_reference")
    def test_from_dict(self, mock_load_from_reference, mock_ref_from_dict) -> None:
        """Test deserialization from dictionary."""
        # Mock factory to return our mock model service
        mock_load_from_reference.return_value = self.mock_model_service

        # Create serialized data
        serialized_data = {
            "embedding_function_type": "model-service",
            "model_service": {"name": "test-model-service"},
            "timeout": 10.0,
            "model": "test-model",
            "embedder_kwargs": {"normalize": True},
        }

        # Deserialize
        embedding_fn: ModelServiceEmbeddingFunction = ModelServiceEmbeddingFunction.from_dict(serialized_data)

        # Check the deserialized embedding function has correct attributes
        self.assertEqual(self.mock_model_service, embedding_fn.model_service)
        self.assertEqual(10.0, embedding_fn.timeout)
        self.assertEqual("test-model", embedding_fn.model)
        self.assertEqual({"normalize": True}, embedding_fn.embedder_kwargs)

        # Verify factory methods was called with correct reference
        mock_ref_from_dict.assert_called_once_with({"name": "test-model-service"})
        mock_load_from_reference.assert_called_once_with(mock_ref_from_dict.return_value)

    @patch("ravelin.embedding_functions.model_service.AssetReference.from_dict")
    @patch("ravelin.embedding_functions.model_service.ModelServiceFactory.load_from_reference")
    def test_from_dict_with_invalid_model_service(self, mock_load_from_reference, _) -> None:
        """Test deserialization fails with invalid model service type."""
        # Return a non-EmbeddingModelService
        mock_load_from_reference.return_value = MagicMock()

        serialized_data = {
            "embedding_function_type": "model-service",
            "model_service": {"name": "test-model-service"},
        }

        # Should raise TypeError
        with self.assertRaisesRegex(TypeError, "The model service must be an instance of EmbeddingModelService"):
            ModelServiceEmbeddingFunction.from_dict(serialized_data)


if __name__ == "__main__":
    unittest.main()
