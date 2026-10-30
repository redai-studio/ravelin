"""Model service embedding function implementation.

This module provides a DenseEmbeddingFunction implementation that uses an
EmbeddingModelService to generate dense embeddings from text or other inputs.
"""

from typing import Any, Dict, List, Optional, TypeVar

from typing_extensions import Self

from ravelin.assets.asset_factory import ModelServiceFactory
from ravelin.assets.model_services import EmbeddingModelService
from ravelin.dataclasses import AssetReference
from ravelin.embedding_functions._base import (
    RegisteredEmbeddingFunction,
    register_embedding_function,
)

T = TypeVar("T")


@register_embedding_function("model-service")
class ModelServiceEmbeddingFunction(RegisteredEmbeddingFunction):
    """Dense embedding function implementation using an EmbeddingModelService.

    This class wraps an EmbeddingModelService instance to create a DenseEmbeddingFunction
    that can generate embeddings for text or other input types.

    :param model_service: The EmbeddingModelService to use for embedding
    :param timeout: Optional timeout for embedding requests (in seconds)
    :param model: Optional model identifier to use within the service
    :param embedder_kwargs: Optional keyword arguments to pass to the embedding service
    """

    def __init__(
        self,
        model_service: EmbeddingModelService,
        timeout: Optional[float] = None,
        model: Optional[str] = None,
        embedder_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_service = model_service
        self.timeout = timeout
        self.model = model
        self.embedder_kwargs = embedder_kwargs or {}
        super().__init__()

    def __call__(self, inputs: List[T]) -> List[List[float]]:
        """Generate embeddings for the given inputs.

        :param inputs: The inputs to embed. Can be a list of strings, documents, or other supported types
                      depending on the underlying model service.
        :return: A list of dense embedding vectors, where each vector is a list of floats.
        """
        # Call the model service to generate embeddings
        return self.model_service.infer(
            inputs=inputs, timeout=self.timeout, model=self.model, inference_kwargs=self.embedder_kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the embedding function to a serializable dictionary.

        :return: A dictionary representing this embedding function that can be
                used to reconstruct it.
        """
        return {
            "embedding_function_type": self.embedding_function_type,
            "model_service": self.model_service.reference.to_dict(),
            "timeout": self.timeout,
            "model": self.model,
            "embedder_kwargs": self.embedder_kwargs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """Create an embedding function from a serialization dictionary.

        :param data: A dictionary containing the information needed to reconstruct
                    the embedding function.
        :return: The reconstructed embedding function.
        """
        # Load the model service from reference
        model_service_reference = AssetReference.from_dict(data["model_service"])
        model_service = ModelServiceFactory.load_from_reference(model_service_reference)

        if not isinstance(model_service, EmbeddingModelService):
            raise TypeError(
                f"The model service must be an instance of EmbeddingModelService, got {type(model_service)}"
            )

        # Create the embedding function
        return cls(
            model_service=model_service,
            timeout=data.get("timeout"),
            model=data.get("model"),
            embedder_kwargs=data.get("embedder_kwargs", {}),
        )
