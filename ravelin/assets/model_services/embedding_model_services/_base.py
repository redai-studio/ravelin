"""
This module defines the base class for embedding model services. An embedding model service is a deployed model (or a
selection of models) that can be used for generating embeddings from input data, such as text or images. Embeddings are
vector representations of the input data that can be used for various downstream tasks, such as similarity search or
input to other models.
"""

import abc
from typing import Any, Dict, List, Mapping, Optional

import httpx

from ravelin.assets._base import SecretValue
from ravelin.assets._httpx import _HttpxClientMixin
from ravelin.assets.model_services._base import _HttpModelService, _ModelService


class _EmbeddingModelService(_ModelService, metaclass=abc.ABCMeta):
    """
    An abstract base class of a generic service for managing and interacting with embedding models.
    """

    @abc.abstractmethod
    # pylint: disable=arguments-differ
    def infer(
        self,
        inputs: List[Any],
        *,
        model: Optional[str],
        timeout: Optional[float],
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for the given query using the model service.

        :param inputs: The inputs to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :return: The list of embeddings.
        """
        raise NotImplementedError("infer method must be implemented")


class _HttpxEmbeddingModelService(_HttpxClientMixin, _HttpModelService, _EmbeddingModelService, metaclass=abc.ABCMeta):
    """
    An abstract base class of a generic service for managing and interacting with embedding models using HTTPX.

    :param name: The name of the model service.
    :param api_key: The API key for the model service as a Haystack ``Secret``.
    :param base_url: The base URL of the model service.
    :param organization: The organization for the model service.
    :param routes: Per-instance operation-route overrides.
    :param client_kwargs: Additional keyword arguments to pass to the HTTPX client.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: SecretValue = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        routes: Optional[Mapping[str, str]] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            routes=routes,
        )
        if client_kwargs:
            self.configs["client_kwargs"] = client_kwargs

    @property
    def client_kwargs(self) -> Dict[str, Any]:
        """
        Get the client keyword arguments for the HTTPX client.

        :return: The client keyword arguments.
        """
        return self.configs.get("client_kwargs", {})

    @abc.abstractmethod
    def build_request(
        self,
        client: httpx.Client,
        inputs: List[Any],
        *,
        model: Optional[str] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> httpx.Request:
        """
        Build a request according to the corresponding embedding API.

        :param client: The httpx client to use for the request.
        :param inputs: The inputs to send to the model.
        :param model: The model to use for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :param kwargs: Additional keyword arguments to pass to the httpx client.
        :return: The httpx request object.
        """
        raise NotImplementedError("build_request method must be implemented")

    @abc.abstractmethod
    def parse_response(
        self,
        response: httpx.Response,
    ) -> List[List[float]]:
        """
        Parse the response from the corresponding embedding API to a list of embeddings.

        :param response: The httpx response from the model service.
        :return: The list of embeddings.
        """
        raise NotImplementedError("parse_response method must be implemented")

    def infer(
        self,
        inputs: List[Any],
        *,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for the given query using the model service.

        :param inputs: The inputs to send to the model.
        :param model: The model to use for the request.
        :param timeout: The timeout for the request.
        :param inference_kwargs: Additional generation keyword arguments to include in the request.
        :return: The list of embeddings.
        """
        client = self._client
        request = self.build_request(
            client,
            inputs,
            model=model,
            inference_kwargs=inference_kwargs,
        )
        response = client.send(self._apply_timeout(request, timeout)).raise_for_status()
        return self.parse_response(response)
