"""
This module defines JinaEmbeddingModelService, a ModelService asset that refers to a Jina embedding model.
"""

from types import MappingProxyType
from typing import Optional, List, Dict, Any

import httpx

from ravelin.assets._base import resolve_secret
from ravelin.assets.model_services._base import register_model_service
from ravelin.assets.model_services.embedding_model_services._base import (
    _HttpxEmbeddingModelService,
)


@register_model_service("jina-embedding")
class JinaEmbeddingModelService(_HttpxEmbeddingModelService):
    """
    A ModelService asset that refers to a Jina embedding model.
    """

    DEFAULT_ROUTES = MappingProxyType({"inference": "/v1/embeddings"})

    def build_request(
        self,
        client: httpx.Client,
        inputs: List[Any],
        model: Optional[str] = None,
        *,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Request:
        """
        Build a request according to the Jina API format.

        :param client: The httpx client to use for the request.
        :param inputs: The inputs to send to the model.
        :param model: The model to use for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param kwargs: Additional keyword arguments to pass to the httpx client.
        :return: The httpx request object.
        """
        model = model or "jina-embeddings-v2-base-zh"
        inference_kwargs = inference_kwargs or {}
        headers = {
            "Content-Type": "application/json",
        }
        api_key = resolve_secret(self.api_key)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {
            "model": model,
            "input": inputs,
            **inference_kwargs,
        }

        # Build the request
        request = client.build_request(
            "POST",
            url=self.build_request_url("inference"),
            json=body,
            headers=headers,
            **kwargs,
        )

        return request

    def parse_response(
        self,
        response: httpx.Response,
    ) -> List[List[float]]:
        """
        Parse the response from the Jina API to a list of embeddings.

        :param response: The response body from the Jina API.
        :return: The list of embeddings.
        """
        response_body = response.json()
        data = response_body.get("data", [])
        return [entry.get("embedding") for entry in data]
