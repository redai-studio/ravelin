"""
This module contains the dataclass for EmbeddingFields.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

from typing_extensions import Self

from ravelin.dataclasses.asset_reference import AssetReference

if TYPE_CHECKING:
    from ravelin.embedding_functions import RegisteredEmbeddingFunction


class VectorType(str, Enum):
    """
    Enum representing the type of vector embedding.

    :cvar DENSE: Dense vector embeddings (default for most embedding models)
    :cvar SPARSE: Sparse vector embeddings (e.g., BM25, TFIDF)
    """

    DENSE = "dense"
    SPARSE = "sparse"


class _BackwardCompatible(type):
    """
    A metaclass for backward compatibility with older embedding fields.
    """

    def __call__(cls, *args, **kwargs):
        """
        Called before EmbeddingField.__init__, handles legacy fields.
        """
        if "from_asset" in kwargs and "embedding_function_data" not in kwargs:
            # Convert legacy fields to serialized embedding function data
            from_asset = kwargs.pop("from_asset")
            embedder_kwargs = kwargs.pop("embedder_kwargs", {}) or {}

            if from_asset is not None:
                kwargs["_embedding_function_data"] = {
                    "embedding_function_type": "model-service",
                    "model_service": from_asset.to_dict(),
                    "timeout": embedder_kwargs.pop("timeout", None),
                    "model": embedder_kwargs.pop("model", None),
                    "embedder_kwargs": embedder_kwargs,
                }
        return super().__call__(*args, **kwargs)


@dataclass
class EmbeddingField(metaclass=_BackwardCompatible):
    """
    A dataclass that represents the embedder and search metadata related to embedding fields in the vector database.

    :param search_kwargs: Additional arguments for vector search
    :param vector_type: The type of vector embedding (dense or sparse)
    """

    _embedding_function_data: Optional[Dict[str, Any]] = None
    """
    Serialized data for the embedding function (for internal use)
    """

    search_kwargs: Optional[Dict[str, Any]] = None
    vector_type: VectorType = VectorType.DENSE
    _loaded_embedding_function: Optional["RegisteredEmbeddingFunction"] = None

    @property
    def embedding_function(self) -> Optional["RegisteredEmbeddingFunction"]:
        """
        Get the embedding function, loading it from serialized data if needed.

        :return: The embedding function or None if not configured
        """
        if self._loaded_embedding_function is None and self._embedding_function_data is not None:
            # Import at runtime to avoid circular imports
            # pylint: disable=import-outside-toplevel
            from ravelin.embedding_functions import EmbeddingFunctionFactory

            self._loaded_embedding_function = EmbeddingFunctionFactory.create_from_dict(self._embedding_function_data)
        return self._loaded_embedding_function

    @embedding_function.setter
    def embedding_function(self, value: Optional["RegisteredEmbeddingFunction"]) -> None:
        """
        Set the embedding function, updating the serialized data.

        :param value: The embedding function to set
        """
        self._loaded_embedding_function = value
        if value is not None:
            self._embedding_function_data = value.to_dict()
        else:
            self._embedding_function_data = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a serialized dictionary representation of the dataclass.

        :returns: A dictionary representation of the dataclass
        """
        data = {
            "search_kwargs": self.search_kwargs,
            "vector_type": self.vector_type.value,
        }

        # Store the serialized embedding function data
        if self._embedding_function_data is not None:
            data["embedding_function"] = self._embedding_function_data

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Creates an instance of the dataclass from a dictionary.

        :param data: A dictionary containing the dataclass attributes
        :return: An instance of the dataclass
        """
        vector_type = data.get("vector_type", VectorType.DENSE.value)
        if isinstance(vector_type, str):
            vector_type = VectorType(vector_type)

        # Store the serialized data for lazy loading
        _embedding_function_data = None
        if "embedding_function" in data:
            _embedding_function_data = data["embedding_function"]
        elif "from_asset" in data:
            # Handle legacy format
            from_asset_data = AssetReference.from_dict(data["from_asset"])
            embedder_kwargs = data.get("embedder_kwargs", {}) or {}

            _embedding_function_data = {
                "embedding_function_type": "model-service",
                "model_service": from_asset_data.to_dict(),
                "timeout": embedder_kwargs.pop("timeout", None),
                "model": embedder_kwargs.pop("model", None),
                "embedder_kwargs": embedder_kwargs,
            }

        return cls(
            _embedding_function_data=_embedding_function_data,
            search_kwargs=data.get("search_kwargs"),
            vector_type=vector_type,
        )
