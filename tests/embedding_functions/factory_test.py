"""Unit tests for EmbeddingFunctionFactory."""

import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from typing_extensions import Self

from ravelin.embedding_functions import (
    EmbeddingFunctionFactory,
    RegisteredEmbeddingFunction,
)
from ravelin.embedding_functions._base import embedding_function_registry


# Create a mock embedding function class that can be registered
class MockEmbeddingFunction(RegisteredEmbeddingFunction):
    """Mock embedding function for testing."""

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        """Mock implementation."""
        return [[1.0, 2.0, 3.0] for _ in inputs]

    def to_dict(self) -> Dict[str, Any]:
        """Mock implementation."""
        return {"embedding_function_type": "test_type"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Mock implementation."""
        # Create a mock instance that will be returned when from_dict is called
        return cls()


class EmbeddingFunctionFactoryTest(unittest.TestCase):
    """Test suite for EmbeddingFunctionFactory."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Save the original registry, so we can restore it after tests
        self.original_registry = embedding_function_registry.get_all()

        # Register our mock class
        embedding_function_registry.register("test_type", MockEmbeddingFunction)

        # Create a mock instance for test_create_from_dict_valid test
        self.mock_instance = MagicMock(spec=RegisteredEmbeddingFunction)

        # Patch the from_dict method to return our mock instance
        self.original_from_dict = MockEmbeddingFunction.from_dict
        MockEmbeddingFunction.from_dict = MagicMock(return_value=self.mock_instance)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        # Restore the original registry
        embedding_function_registry._registry = self.original_registry

        # Restore the original from_dict method
        MockEmbeddingFunction.from_dict = self.original_from_dict

    def test_create_from_dict_valid(self) -> None:
        """Test creating an embedding function from a valid serialization dict."""
        # Test data
        serialized_data = {"embedding_function_type": "test_type", "some_param": "test_value"}

        # Create from dict
        result = EmbeddingFunctionFactory.create_from_dict(serialized_data)

        # Verify the mock class was called with the serialized data
        MockEmbeddingFunction.from_dict.assert_called_once_with(serialized_data)

        # Verify the result is the mock instance
        self.assertEqual(self.mock_instance, result)

    def test_create_from_dict_missing_type(self) -> None:
        """Test that an error is raised when the type field is missing."""
        serialized_data = {"some_param": "test_value"}

        with self.assertRaises(ValueError) as context:
            EmbeddingFunctionFactory.create_from_dict(serialized_data)

        self.assertIn("Missing 'embedding_function_type' field", str(context.exception))

    def test_create_from_dict_invalid_type(self) -> None:
        """Test that an error is raised for an unknown embedding function type."""
        serialized_data = {"embedding_function_type": "nonexistent_type", "some_param": "test_value"}

        with self.assertRaisesRegex(ValueError, "Unknown embedding function type: 'nonexistent_type'"):
            EmbeddingFunctionFactory.create_from_dict(serialized_data)

    def test_create_from_dict_empty_registry(self) -> None:
        """Test error message when registry is empty."""
        # Clear the registry
        embedding_function_registry._registry = {}

        serialized_data = {
            "embedding_function_type": "nonexistent_type",
        }

        with self.assertRaises(ValueError) as context:
            EmbeddingFunctionFactory.create_from_dict(serialized_data)

        self.assertIn("No types registered", str(context.exception))


if __name__ == "__main__":
    unittest.main()
