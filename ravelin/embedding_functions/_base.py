"""
This module defines protocols, base classes and a registry for embedding functions.

Protocols define the expected behavior, while RegisteredEmbeddingFunction provides
registration and serialization support for functions that need it.

The embedding_function_registry is a singleton registry that stores all registered
embedding functions. It's used to look up embedding functions by type and to
register new ones.

The register_embedding_function decorator is used to register new embedding functions
with the registry.
"""

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, TypeVar, Union, runtime_checkable

from haystack.dataclasses import SparseEmbedding
from typing_extensions import Self

from ravelin.utils.registry import RegistryMeta, Registry

T = TypeVar("T")  # Generic type for inputs

embedding_function_registry = Registry["RegisteredEmbeddingFunction"]("embedding_function_type")
register_embedding_function = embedding_function_registry.create_decorator()


@runtime_checkable
class DenseEmbeddingFunction(Protocol):
    """Protocol for dense embedding functions.

    This protocol defines the minimal interface for functions that generate
    dense embeddings from various input types. Implementations should be callable
    objects that convert input text or other data into dense vector representations.

    Example:
        >>> embedder = MyEmbeddingFunction()
        >>> embedding = embedder("text to embed")
        >>> embeddings = embedder(["multiple", "texts", "to embed"])

    Note:
        Any function that accepts a list of inputs and returns a list of dense embeddings
        can be used as a DenseEmbeddingFunction, even if it doesn't formally implement
        this protocol. Registration is only needed for serialization/deserialization.
    """

    def __call__(self, inputs: List[T]) -> List[List[float]]:
        """Generate embeddings for the given inputs.

        :param inputs: The inputs to embed. Can be a list of strings, documents, or other supported types.
        :return: The embeddings for the inputs.
        """


@runtime_checkable
class SparseEmbeddingFunction(Protocol):
    """Protocol for sparse embedding functions that require fitting.

    This extends the basic EmbeddingFunction protocol with methods for
    stateful sparse embedding functions that need to be fitted to a corpus
    before they can generate embeddings.

    Example:
        >>> embedder = MySparseEmbeddingFunction()
        >>> embedder.fit(["training", "corpus", "documents"])
        >>> assert embedder.is_fitted
        >>> embedding = embedder("query text")

    Note:
        Registration is only needed for serialization/deserialization.
    """

    @property
    def is_fitted(self) -> bool:
        """Check if the embedding function is fitted to a corpus.

        :return: True if the embedding function is fitted, False otherwise.
        """

    def fit(
        self, index_path: Optional[Union[str, Path]] = None, corpus: Optional[List[Union[str, T]]] = None
    ) -> "SparseEmbeddingFunction":
        """Fit the embedding function to a corpus.

        :param index_path: Path to a pre-computed index file to load instead of fitting on a corpus.
        :param corpus: The corpus to fit the embedding function to.
        :return: The fitted embedding function (self).
        """

    def __call__(self, inputs: List[T]) -> List[SparseEmbedding]:
        """Generate embeddings for the given inputs.

        :param inputs: The input(s) to embed. Can be a list of strings, documents, or other supported types.
        :return: The embeddings for the inputs.
        """


# Union type for type hints
EmbeddingFunction = Union[DenseEmbeddingFunction, SparseEmbeddingFunction]


class RegisteredEmbeddingFunction(metaclass=RegistryMeta):
    """Base class for embedding functions that need registration and serialization.

    This base class is required for embedding functions that need to be:
    1. Serialized/deserialized (saved and loaded)
    2. Created dynamically from serialized data

    It uses AssetRegistryMeta to enforce registration of concrete subclasses,
    ensuring they can be properly reconstructed from serialized data.

    Note:
        Not all embedding functions need to inherit from this class.
        Only those that require serialization support should inherit from it.
    """

    # Type identifier name for the registry metaclass
    TYPE_IDENTIFIER = "embedding_function_type"

    def __init__(self, *args, **kwargs):
        """Initialize the embedding function and verify it implements the EmbeddingFunction protocol.

        This checks at runtime that any registered embedding function actually implements
        either the DenseEmbeddingFunction or SparseEmbeddingFunction protocol.

        :raises TypeError: If the instance doesn't implement the EmbeddingFunction protocol
        """
        if not isinstance(self, DenseEmbeddingFunction) and not isinstance(self, SparseEmbeddingFunction):
            raise TypeError(
                f"{self.__class__.__name__} must implement either DenseEmbeddingFunction or "
                f"SparseEmbeddingFunction protocol."
            )
        super().__init__(*args, **kwargs)

    @property
    def embedding_function_type(self) -> str:
        """Get the type of the embedding function."""
        return getattr(self, f"_{self.__class__.__name__}__{self.TYPE_IDENTIFIER}")

    @abc.abstractmethod
    def __call__(self, inputs: List[T]) -> Any:
        """Generate embeddings for the given inputs.

        All embedding functions must be callable and generate embeddings for their inputs.

        :param inputs: The inputs to embed. Can be a list of strings, documents, or other supported types.
        :return: The embeddings for the inputs, either dense or sparse.
        """

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the embedding function to a serializable dictionary.

        :return: A dictionary representing this embedding function that can be
                used to reconstruct it.
        """

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create an embedding function from a serialization dictionary.

        :param data: A dictionary containing the information needed to reconstruct
                    the embedding function.
        :return: The reconstructed embedding function.
        """
