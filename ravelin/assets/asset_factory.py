"""
This module contains factory classes for creating and loading assets.

The assets can be of different types such as ModelService, VectorDatabase and EmbeddingFunction. Each asset type has
its own factory class (ModelServiceFactory, VectorDatabaseFactory and EmbeddingFunctionFactory respectively) that is
responsible for creating and loading instances of that asset type.

The AssetFactory class is a general factory class that can create and load any type of asset. It uses the specific
factory classes for each asset type to do the actual creation and loading.

This module also provides a way to register new types of ModelService, VectorDatabase and EmbeddingFunction assets,
so that they can be created and loaded by the factory classes.
"""

import abc
from typing import Any, Dict, Generic, TypeVar

from ravelin.assets import Asset
from ravelin.assets.resolvers import LOCAL_ASSET_SOURCE, resolve_asset
from ravelin.assets.model_services import ModelService, MODEL_SERVICE_REGISTRY
from ravelin.assets.tool_services import ToolService, TOOL_SERVICE_REGISTRY
from ravelin.assets.vector_databases import VectorDatabase, VECTOR_DATABASE_REGISTRY
from ravelin.dataclasses.asset_reference import AssetReference

AssetT = TypeVar("AssetT", bound=Asset)


def camel_to_hyphen(s: str) -> str:
    """
    Convert an upper camel case string to a hyphenated string. This is to ensure backward compatibility with the changes
    made to asset type names.

    :param s: The camel case string.
    :return: The hyphenated string.
    """
    return "".join(["-" + c.lower() if c.isupper() else c for c in s]).lstrip("-")


class _BaseFactory(Generic[AssetT], metaclass=abc.ABCMeta):
    """
    Base factory class for creating assets.
    """

    @classmethod
    @abc.abstractmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> AssetT:
        """
        Load an asset from a dictionary.

        :param data: The dictionary containing the asset.
        :return: An asset.
        :raises ValueError: If the asset type is not supported.
        """

    @classmethod
    def load_from_local(cls, asset_name: str) -> AssetT:
        """
        Load an asset from the local storage.

        :param asset_name: The name of the asset to load.
        :return: An asset.
        """
        return cls.load_from_reference(AssetReference(name=asset_name, source=LOCAL_ASSET_SOURCE))

    @classmethod
    def load_from_reference(cls, asset_reference: AssetReference) -> AssetT:
        """
        Load an asset from storage using an asset reference.

        :param asset_reference: The asset reference to the asset.
        :return: An asset.
        """
        asset = cls.load_from_dict(resolve_asset(asset_reference))
        asset.reference = asset_reference
        return asset


class ModelServiceFactory(_BaseFactory[ModelService]):
    """
    Factory class to create ModelService assets.
    """

    @classmethod
    def create(cls, inference_api_format: str, **kwargs) -> ModelService:
        """
        Create a ModelService asset from the given inference API format.

        :param inference_api_format: The inference API format of the model service.
            It must be one of the supported formats, or a ValueError will be raised.
        :param kwargs: Additional keyword arguments to initialize the model service.
        :return: A ModelService asset.
        :raises ValueError: If the inference API format is not supported.
        """
        try:
            model_service_cls = MODEL_SERVICE_REGISTRY.get(inference_api_format)
            return model_service_cls(**kwargs)
        except KeyError as e:
            raise ValueError(f"Unsupported inference_api_format: {inference_api_format}") from e

    @classmethod
    def load_from_dict(cls, data: dict) -> ModelService:
        """
        Load a ModelService asset from a dictionary.

        :param data: The dictionary containing the ModelService asset. It must contain an "inference_api_format" key
            and the key must be one of the supported formats, or a ValueError will be raised.
        :return: A ModelService asset.
        :raises ValueError: If the dictionary does not represent a ModelService asset, or the inference API format is
            not supported.
        """
        if camel_to_hyphen(data.get("type", "")) != "model-service":
            raise ValueError("The dictionary does not represent a ModelService asset.")
        try:
            configs_data = data.get("configs", {}).copy()
            inference_api_format = configs_data.pop("inference_api_format")
        except KeyError as err:
            raise ValueError("Missing 'inference_api_format' key in the dictionary") from err
        try:
            model_service_cls = MODEL_SERVICE_REGISTRY.get(inference_api_format)
            return model_service_cls.from_dict(
                {
                    "name": data.get("name"),
                    "configs": configs_data,
                    "secrets": data.get("secrets", {}),
                }
            )
        except KeyError as err:
            raise ValueError(f"Unsupported inference_api_format: {inference_api_format}") from err


class ToolServiceFactory(_BaseFactory[ToolService]):
    """
    Factory class to create ToolService assets.
    """

    @classmethod
    def create(cls, call_api_format: str, **kwargs) -> ToolService:
        """
        Create a ToolService asset from the given call API format.

        :param call_api_format: The call API format of the tool service.
            It must be one of the supported formats, or a ValueError will be raised.
        :param kwargs: Additional keyword arguments to initialize the tool service.
        :return: A ToolService asset.
        :raises ValueError: If the call API format is not supported.
        """
        try:
            tool_service_cls = TOOL_SERVICE_REGISTRY.get(call_api_format)
            return tool_service_cls(**kwargs)
        except KeyError as e:
            raise ValueError(f"Unsupported call_api_format: {call_api_format}") from e

    @classmethod
    def load_from_dict(cls, data: dict) -> ToolService:
        """
        Load a ToolService asset from a dictionary.

        :param data: The dictionary containing the ToolService asset. It must contain a "call_api_format" key
            and the key must be one of the supported formats, or a ValueError will be raised.
        :return: A ToolService asset.
        :raises ValueError: If the dictionary does not represent a ToolService asset, or the inference API format is
            not supported.
        """
        if camel_to_hyphen(data.get("type", "")) != "tool-service":
            raise ValueError("The dictionary does not represent a ToolService asset.")
        try:
            configs_data = data.get("configs", {}).copy()
            call_api_format = configs_data.pop("call_api_format")
        except KeyError as err:
            raise ValueError("Missing 'call_api_format' key in the dictionary") from err
        try:
            tool_service_cls = TOOL_SERVICE_REGISTRY.get(call_api_format)
            return tool_service_cls.from_dict(
                {
                    "name": data.get("name"),
                    "configs": configs_data,
                    "secrets": data.get("secrets", {}),
                }
            )
        except KeyError as err:
            raise ValueError(f"Unsupported call_api_format: {call_api_format}") from err


class VectorDatabaseFactory(_BaseFactory[VectorDatabase]):
    """
    Factory class to create VectorDatabase assets.
    """

    @classmethod
    def create(cls, vector_db_type: str, **kwargs) -> VectorDatabase:
        """
        Create a VectorDatabase asset from the given vector database type.

        :param vector_db_type: The type of the vector database.
            It must be one of the supported types, or a ValueError will be raised.
        :param kwargs: Additional keyword arguments to initialize the vector database.
        :return: A VectorDatabase asset.
        :raises ValueError: If the vector database type is not supported.
        """
        try:
            vector_database_cls = VECTOR_DATABASE_REGISTRY.get(vector_db_type)
            return vector_database_cls(**kwargs)
        except KeyError as e:
            raise ValueError(f"Unsupported vector_db_type: {vector_db_type}") from e

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> VectorDatabase:
        """
        Load a VectorDatabase asset from a dictionary.

        :param data: The dictionary containing the VectorDatabase asset. It must contain a "vector_db_type" key
            and the key must be one of the supported types, or a ValueError will be raised.
        :return: A VectorDatabase asset.
        :raises ValueError: If the vector database type is not supported.
        """
        if camel_to_hyphen(data.get("type", "")) != "vector-database":
            raise ValueError("The dictionary does not represent a VectorDatabase asset.")
        try:
            configs_data = data.get("configs", {}).copy()
            vector_db_type = configs_data.pop("vector_db_type")
        except KeyError as err:
            raise ValueError("Missing 'vector_db_type' key in the dictionary") from err
        try:
            vector_database_cls = VECTOR_DATABASE_REGISTRY.get(vector_db_type)
            return vector_database_cls.from_dict(
                {
                    "name": data.get("name"),
                    "configs": configs_data,
                    "secrets": data.get("secrets", {}),
                }
            )
        except KeyError as err:
            raise ValueError(f"Unsupported vector_db_type: {vector_db_type}") from err


class AssetFactory(_BaseFactory[Asset]):
    """
    Factory class to create assets.
    """

    @staticmethod
    def create(asset_type: str, **kwargs) -> Asset:
        """
        Create an asset from the given type.

        :param asset_type: The type of the asset to create.
        :param kwargs: Additional keyword arguments to initialize the asset.
        :return: An asset.
        :raises ValueError: If the asset type is not supported.
        """
        if camel_to_hyphen(asset_type) == "model-service":
            return ModelServiceFactory.create(**kwargs)
        if camel_to_hyphen(asset_type) == "tool-service":
            return ToolServiceFactory.create(**kwargs)
        if camel_to_hyphen(asset_type) == "vector-database":
            return VectorDatabaseFactory.create(**kwargs)
        # Add more conditions here to handle future types of assets
        raise ValueError(f"Unsupported asset_type: {asset_type}")

    @classmethod
    def load_from_dict(cls, data: Dict[str, Any]) -> Asset:
        """
        Load an asset from a dictionary.

        :param data: The dictionary containing the asset.
        :return: An asset.
        :raises ValueError: If the asset type is not supported.
        """
        asset_type = data.get("type", "")
        if camel_to_hyphen(asset_type) == "model-service":
            return ModelServiceFactory.load_from_dict(data)
        if camel_to_hyphen(asset_type) == "tool-service":
            return ToolServiceFactory.load_from_dict(data)
        if camel_to_hyphen(asset_type) == "vector-database":
            return VectorDatabaseFactory.load_from_dict(data)
        # Add more conditions here to handle future types of assets
        raise ValueError(f"Unsupported asset_type: {asset_type}")
