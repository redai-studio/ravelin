"""
This module defines the OpenAIEmbeddingModelService class for handling the OpenAI embedding model service or any model
supporting the OpenAI Embedding API format.
"""

import logging
from types import MappingProxyType
from typing import Optional, List, Dict, Any

import httpx

from ravelin.assets._base import resolve_secret
from ravelin.assets.model_services._base import register_model_service
from ravelin.assets.model_services.embedding_model_services._base import (
    _HttpxEmbeddingModelService,
)

logger = logging.getLogger(__name__)


@register_model_service("openai-embedding")
class OpenAIEmbeddingModelService(_HttpxEmbeddingModelService):
    """
    A ModelService asset that refers to a deployed OpenAI model, or a deployed model supporting OpenAI Embedding API
    format.
    """

    DEFAULT_ROUTES = MappingProxyType({"inference": "/v1/embeddings"})

    def build_request(
        self,
        client: httpx.Client,
        inputs: List[Any],
        *,
        model: Optional[str] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Request:
        model = model or "text-embedding-3-large"
        inference_kwargs = inference_kwargs or {}

        for i in inputs:
            if not isinstance(i, str):
                raise ValueError(f"Input {i} is not a string and cannot be embedded in OpenAI Embedding API format")

        body = {
            "input": inputs,
            "model": model,
            **inference_kwargs,
        }

        # Create the headers for the request
        headers = {
            "Content-Type": "application/json",
        }
        api_key = resolve_secret(self.api_key)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            headers["api-key"] = api_key  # For Azure OpenAI API compatibility
            if self.organization:
                headers["OpenAI-Organization"] = self.organization

        # Build the request
        request = client.build_request(
            "POST",
            url=self.build_request_url("inference"),
            json=body,
            headers=headers,
            **kwargs,
        )
        return request

    def parse_response(self, response: httpx.Response) -> List[List[float]]:
        response_body = response.json()
        data = response_body.get("data", [])
        return [entry.get("embedding") for entry in data]
