"""
This module provides a factory for loading embedding function implementations
from serialized dictionaries. It's used for reconstructing registered embedding
functions that have been previously serialized.
"""

from typing import Any, Dict

from ravelin.embedding_functions._base import RegisteredEmbeddingFunction, embedding_function_registry


class EmbeddingFunctionFactory:
    """
    Factory class for creating and loading registered embedding functions.

    This factory uses the embedding function registry to create instances
    of registered embedding function implementations from serialized data.
    Only embedding functions that have been registered with the register_embedding_function
    decorator can be reconstructed using this factory.
    """

    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> RegisteredEmbeddingFunction:
        """
        Create an embedding function from a serialized dictionary.

        :param data: A dictionary containing the serialized embedding function
        :return: An instantiated embedding function
        :raises ValueError: If the type is missing or not registered
        """
        if "embedding_function_type" not in data:
            raise ValueError("Missing 'embedding_function_type' field in serialization data")

        type_identifier = data["embedding_function_type"]
        try:
            # Get the class from the registry and create the instance
            embedding_function_cls = embedding_function_registry.get(type_identifier)
            return embedding_function_cls.from_dict(data)
        except KeyError as e:
            registered_types = ", ".join(sorted(embedding_function_registry.get_all().keys()))
            available_msg = (
                f"Available types: {registered_types}"
                if embedding_function_registry.get_all()
                else "No types registered."
            )
            raise ValueError(f"Unknown embedding function type: '{type_identifier}'. {available_msg}") from e
