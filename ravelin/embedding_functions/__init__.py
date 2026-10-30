"""
The embedding functions module provides the core architecture for embedding functions in the ravelin library,
including abstractions, preset embedding functions and registry & factory classes for embedding functions.

The goal is to provide a flexible and extensible framework for embedding functions for vector database searches and
embedding-related pipeline components.

The main components are:

1. Protocols (DenseEmbeddingFunction, SparseEmbeddingFunction):
   - Define the expected behavior of embedding functions
   - Used for type checking and interface documentation
   - Any callable that follows these patterns can be used as an embedding function

2. RegisteredEmbeddingFunction:
   - Base class for embedding functions that need serialization/deserialization
   - Provides registration through the AssetRegistryMeta metaclass
   - Required if the embedding function needs to be serialized/deserialized (alone or in a pipeline)

3. Preset Embedding Functions:
   - Preset RegisteredEmbeddingFunction subclasses for common embedding functions
   - Provides a default implementation of the embedding function and its serialization

4. Registry and Factory:
   - Decorators for registering serializable embedding functions
   - Factory class for creating any registered embedding function from a dictionary
"""

from ravelin.embedding_functions._base import (
    DenseEmbeddingFunction,
    EmbeddingFunction,
    RegisteredEmbeddingFunction,
    SparseEmbeddingFunction,
    register_embedding_function,
)
from ravelin.embedding_functions.factory import EmbeddingFunctionFactory
from ravelin.embedding_functions.model_service import ModelServiceEmbeddingFunction

__all__ = [
    # protocols
    "EmbeddingFunction",
    "DenseEmbeddingFunction",
    "SparseEmbeddingFunction",
    # base class
    "RegisteredEmbeddingFunction",
    # registry & factory
    "EmbeddingFunctionFactory",
    "register_embedding_function",
    # concrete embedding functions
    "ModelServiceEmbeddingFunction",
]
