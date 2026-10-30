"""
This module defines the base class for model services. A model service is a deployed model (or a selection of models)
that can be used for inference. Model services can be used to generate responses to input data, such as text or images,
and have wide-ranging use cases in an application.
"""

import abc
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Optional, Type

from ravelin.assets._base import _Asset, SecretValue
from ravelin.assets.resolvers import normalize_asset_secrets
from ravelin.utils.registry import RegistryMeta, Registry

# Create a registry for model service formats
MODEL_SERVICE_REGISTRY = Registry["_ModelService"]("INFERENCE_API_FORMAT")

# Create decorator for registering model services
register_model_service: Callable[[str], Callable[[Type["_ModelService"]], Type["_ModelService"]]] = (
    MODEL_SERVICE_REGISTRY.create_decorator()
)


class _ModelService(_Asset, metaclass=RegistryMeta):
    """
    A ModelService asset refers to a deployed model (or a selection of models) that can be used for inference.

    :param name: The name of the model service.
    :param api_key: The API key to use for the model service as a Haystack ``Secret``, defaults to None.
    :param organization: The organization of the model service, defaults to None.
    """

    ASSET_TYPE = "model-service"
    TYPE_IDENTIFIER = "INFERENCE_API_FORMAT"

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: SecretValue = None,
        organization: Optional[str] = None,
    ):
        self.organization = organization
        configs = {
            "organization": organization,
            "inference_api_format": self.inference_api_format,
        }
        if api_key is not None:
            secrets = {"api_key": api_key}
        else:
            secrets = {}

        super().__init__(name, configs=configs, secrets=secrets)

    @property
    def inference_api_format(self) -> str:
        """Get the inference api format."""
        return getattr(self, f"_{self.__class__.__name__}__INFERENCE_API_FORMAT")

    @property
    def api_key(self) -> SecretValue:
        """Get the API-key secret; resolve it with ``resolve_secret`` before use."""
        return self.secrets.get("api_key")

    @api_key.setter
    def api_key(self, value: SecretValue) -> None:
        """Set the api key."""
        self.secrets["api_key"] = normalize_asset_secrets({"secrets": {"api_key": value}})["secrets"]["api_key"]

    @abc.abstractmethod
    def infer(self, *args, **kwargs) -> Any:
        """
        Perform inference using the model service. The exact definition and signature of this method will depend on the
        specific model service implementation.

        :param args: positional arguments to pass to the model service.
        :param kwargs: keyword arguments to pass to the model service.
        :return: the result of the inference.
        """
        raise NotImplementedError("The infer method must be implemented by subclasses.")


class _HttpModelService(_ModelService, metaclass=abc.ABCMeta):
    """Base class for model services invoked over HTTP.

    Concrete protocols own their operation routes through ``DEFAULT_ROUTES``. The
    base mapping is intentionally empty: an HTTP service may derive request URLs
    through discovery metadata or override :meth:`build_request_url` entirely.

    :param name: The name of the model service.
    :param api_key: The API key to use for the model service as a Haystack ``Secret``.
    :param base_url: The root URL of the HTTP model service. It may include a
        deployment path prefix.
    :param organization: The organization of the model service, defaults to None.
    :param routes: Per-instance operation-route overrides. These are merged over
        the concrete service's immutable class-level defaults.
    """

    DEFAULT_ROUTES: ClassVar[Mapping[str, str]] = MappingProxyType({})

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        api_key: SecretValue = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        routes: Optional[Mapping[str, str]] = None,
    ):
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("A base URL must be provided to connect to the HTTP model service.")

        super().__init__(name=name, api_key=api_key, organization=organization)
        self.base_url = base_url.rstrip("/")

        route_overrides = dict(routes or {})
        if not all(isinstance(name, str) and isinstance(route, str) for name, route in route_overrides.items()):
            raise TypeError("Model-service routes must map string operation names to string route templates.")

        self._routes = MappingProxyType(dict(self.DEFAULT_ROUTES) | route_overrides)
        self.configs["base_url"] = base_url
        if routes is not None:
            self.configs["routes"] = route_overrides

    @property
    def routes(self) -> Mapping[str, str]:
        """Return the immutable effective operation routes for this service."""
        return self._routes

    def build_request_url(self, operation: str, **route_parameters: str) -> str:
        """Build a request URL from the service base URL and an operation route.

        Concrete services may override this when routing depends on discovery or
        other runtime metadata.

        :param operation: The operation route to resolve.
        :param route_parameters: Values used to expand the route template.
        :return: The absolute request URL.
        :raises ValueError: If the operation is unknown or a template parameter is missing.
        """
        try:
            route_template = self.routes[operation]
        except KeyError as error:
            raise ValueError(f"No route is configured for model-service operation '{operation}'.") from error

        try:
            route = route_template.format(**route_parameters)
        except KeyError as error:
            missing_parameter = error.args[0]
            raise ValueError(
                f"Route for model-service operation '{operation}' requires parameter '{missing_parameter}'."
            ) from error

        if not route:
            return self.base_url
        return f"{self.base_url}/{route.lstrip('/')}"
