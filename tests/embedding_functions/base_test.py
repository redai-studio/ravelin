"""Unit tests for embedding functions base module."""

import unittest
from typing import Any, Dict, List

from typing_extensions import Self

from ravelin.embedding_functions._base import (
    DenseEmbeddingFunction,
    EmbeddingFunction,
    RegisteredEmbeddingFunction,
    SparseEmbeddingFunction,
    embedding_function_registry,
    register_embedding_function,
)


# Create concrete implementations for testing
class SimpleDenseEmbeddingFunction:
    """Simple concrete implementation of DenseEmbeddingFunction."""

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        """Generate dense embeddings for inputs."""
        # Simple implementation that creates a vector of [1.0, 2.0, 3.0] for each input
        return [[1.0, 2.0, 3.0] for _ in inputs]


class SimpleSparseEmbeddingFunction:
    """Simple concrete implementation of SparseEmbeddingFunction."""

    def __init__(self) -> None:
        """Initialize the embedding function."""
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if embedding function is fitted."""
        return self._fitted

    def fit(self, index_path=None, corpus=None) -> "SimpleSparseEmbeddingFunction":
        """Fit the embedding function."""
        self._fitted = True
        return self

    def __call__(self, inputs: List[str]) -> List[Any]:
        """Generate sparse embeddings for inputs."""
        # Simple implementation
        return [{"indices": [0, 1], "values": [1.0, 2.0]} for _ in inputs]


# Test registered implementation
@register_embedding_function("test_embedding")
class TestRegisteredEmbeddingFunction(RegisteredEmbeddingFunction):
    """Test implementation of RegisteredEmbeddingFunction."""

    __test__ = False

    def __init__(self, dimension: int = 3):
        """Initialize the embedding function.

        :param dimension: Dimension of embeddings
        """
        self.dimension = dimension
        super().__init__()

    def __call__(self, inputs: List[str]) -> List[List[float]]:
        """Generate dense embeddings for inputs.

        :param inputs: List of input strings
        :return: List of dense embedding vectors
        """
        return [[1.0] * self.dimension for _ in inputs]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary.

        :return: Serialized representation
        """
        return {"embedding_function_type": "test_embedding", "dimension": self.dimension}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create from serialized dictionary.

        :param data: Serialized data
        :return: Embedding function instance
        """
        return cls(dimension=data.get("dimension", 3))


class EmbeddingFunctionBaseTest(unittest.TestCase):
    """Test suite for embedding function base module."""

    def test_protocol_adherence(self) -> None:
        """Test that concrete implementations adhere to protocols."""
        # Test dense embedding function protocol
        dense_fn = SimpleDenseEmbeddingFunction()
        self.assertIsInstance(dense_fn, DenseEmbeddingFunction)

        # Test sparse embedding function protocol
        sparse_fn = SimpleSparseEmbeddingFunction()
        self.assertIsInstance(sparse_fn, SparseEmbeddingFunction)

        # Test registered function
        registered_fn = TestRegisteredEmbeddingFunction()
        self.assertIsInstance(registered_fn, DenseEmbeddingFunction)
        self.assertIsInstance(registered_fn, RegisteredEmbeddingFunction)

    def test_embedding_function_union_type(self) -> None:
        """Test EmbeddingFunction union type."""
        # Both types should be valid for the union type
        dense_fn = SimpleDenseEmbeddingFunction()
        sparse_fn = SimpleSparseEmbeddingFunction()

        # This is just a type check, if it compiles it works
        functions: List[EmbeddingFunction] = [dense_fn, sparse_fn]
        self.assertEqual(2, len(functions))

    def test_registered_embedding_function_type_validation(self) -> None:
        """Test that RegisteredEmbeddingFunction validates protocol adherence."""

        # Create a class that doesn't adhere to any protocol
        class InvalidEmbeddingFunction(RegisteredEmbeddingFunction):
            """Invalid embedding function that doesn't implement required protocols."""

            def __call__(self, inputs: List[str]) -> List[Any]:
                return []

            def to_dict(self) -> Dict[str, Any]:
                return {}

            @classmethod
            def from_dict(cls, data: Dict[str, Any]) -> Self:
                return cls()

        # Instantiating should raise TypeError
        with self.assertRaises(TypeError):
            InvalidEmbeddingFunction()

    def test_registered_embedding_function_initialization(self) -> None:
        """Test initialization of a registered embedding function."""
        fn: TestRegisteredEmbeddingFunction = TestRegisteredEmbeddingFunction(dimension=5)

        # Check dimension was set correctly
        self.assertEqual(5, fn.dimension)

        # Check the type identifier property works
        self.assertEqual("test_embedding", fn.embedding_function_type)

    def test_to_dict_from_dict_roundtrip(self) -> None:
        """Test serialization and deserialization roundtrip."""
        original: TestRegisteredEmbeddingFunction = TestRegisteredEmbeddingFunction(dimension=7)

        # Serialize
        serialized = original.to_dict()

        # Deserialize
        deserialized: TestRegisteredEmbeddingFunction = TestRegisteredEmbeddingFunction.from_dict(serialized)

        # Check properties match
        self.assertEqual(7, deserialized.dimension)
        self.assertEqual("test_embedding", deserialized.embedding_function_type)

    def test_registry_registration(self) -> None:
        """Test registration with the embedding function registry."""
        # The test_embedding type should be in the registry
        registered_classes = embedding_function_registry.get_all()
        self.assertIn("test_embedding", registered_classes)
        self.assertEqual(TestRegisteredEmbeddingFunction, registered_classes["test_embedding"])


if __name__ == "__main__":
    unittest.main()
