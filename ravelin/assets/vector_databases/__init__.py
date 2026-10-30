"""
This module provides a collection of classes for interacting with different vector databases.

A vector database is a database that stores vector representations of data. Vector databases are widely used as the
backend for retrieval-based applications that require fast and efficient similarity search to find the most relevant
data points, playing a crucial role in Retrieval Augmented Generation (RAG) applications.

Vector databases include:

- VectorDatabase:
    An abstract base class for managing and interacting with generic vector databases.
- MilvusVectorDatabase:
    A class specifically designed for managing and interacting with Milvus vector databases.
"""

from ravelin.assets.vector_databases._base import (
    _VectorDatabase as VectorDatabase,
    VECTOR_DATABASE_REGISTRY,
    register_vector_database,
)
from ravelin.assets.vector_databases.milvus import MilvusVectorDatabase

__all__ = [
    "VectorDatabase",
    "MilvusVectorDatabase",
    "register_vector_database",
    "VECTOR_DATABASE_REGISTRY",
]
