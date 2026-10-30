"""
This module contains the base class for all assets. An asset is a metadata reference to an external service or resource,
which can be utilized across different applications.
"""

import json
import logging
from typing import Dict, Optional, Any

from haystack.utils import Secret
from typing_extensions import Self

from ravelin.dataclasses.asset_reference import AssetReference
from ravelin.assets.resolvers import LOCAL_ASSET_SOURCE, normalize_asset_secrets
from ravelin.utils.asset_storage import get_local_asset_path

logger = logging.getLogger(__name__)

SecretValue = Optional[Secret]
"""The type of a secret value held by an asset: a Haystack ``Secret`` or ``None``."""


def resolve_secret(value: SecretValue) -> Optional[str]:
    """
    Resolve a secret value to a plain string.

    :param value: A Haystack ``Secret`` or ``None``.
    :return: The resolved string value, or ``None`` if no value is set. For a ``Secret``, this defers to
        ``resolve_value()``, which may raise if a strict environment-variable secret is unset.
    """
    if value is None:
        return None
    if not isinstance(value, Secret):
        raise TypeError("Secret values must be Haystack Secret objects.")
    return value.resolve_value()


class _Asset:
    """
    Base class for all assets.
    """

    ASSET_TYPE = "not-set"
    """The type of the asset. This should be overridden by subclasses."""

    _reference: Optional[AssetReference] = None
    """The reference to the asset to keep track how the asset should be loaded."""

    def __init__(
        self,
        name: Optional[str] = None,
        configs: Optional[Dict[str, Any]] = None,
        secrets: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the asset with the given name, configurations, and secrets.

        :param name: The name of the asset
        :param configs: The configurations of the asset
        :param secrets: The secrets of the asset
        """
        self.name = name or ""
        self.configs = configs or {}
        self.secrets = normalize_asset_secrets({"secrets": secrets if secrets is not None else {}})["secrets"]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the asset to a dictionary. Any secret held as a Haystack ``Secret`` is serialized via its own
        ``to_dict()`` (an environment-variable secret becomes a reference; a token secret raises, by design, so its
        value is never persisted). ``None`` values pass through unchanged; plain-string secrets are rejected.

        :return: A dictionary representation of the asset
        """
        normalized = normalize_asset_secrets({"secrets": self.secrets})
        serialized_secrets = {
            key: value.to_dict() if value is not None else None for key, value in normalized["secrets"].items()
        }
        return {
            "name": self.name,
            "type": self.ASSET_TYPE,
            "configs": self.configs,
            "secrets": serialized_secrets,
        }

    @property
    def reference(self) -> AssetReference:
        """
        Get the reference to the asset. If the asset is not loaded from a reference, an error will be raised.

        :return: An AssetReference to the asset
        :raises ValueError: If the asset does not have a corresponding reference
        """
        if not self._reference:
            raise ValueError(f'The asset "{self.name}" is not loaded from a reference and thus cannot be serialized.')
        return self._reference

    @reference.setter
    def reference(self, reference: AssetReference) -> None:
        """
        Set the reference to the asset.

        :param reference: An AssetReference to the asset
        """
        self._reference = reference

    def save_to_local_storage(self, allow_overwrite: bool = False) -> None:
        """
        Save the asset to the local storage. The asset will be saved as a JSON file with the name of the asset.
        The reference of the asset will also be updated accordingly.

        :param allow_overwrite: Whether to allow overwriting the existing asset file. If set to False and the asset file
            already exists, the save operation will be aborted. Default is False.
        """
        path = get_local_asset_path(self.name)
        if path.exists() and not allow_overwrite:
            logger.warning("Asset file %s already exists and allow_overwrite is set to False. Aborting save.", path)
        else:
            logger.info("Saving asset %s to local storage.", self.name)
            serialized = self.to_dict()
            with open(path, "w", encoding="utf-8") as fw:
                json.dump(serialized, fw)
            self.reference = AssetReference(name=self.name, source=LOCAL_ASSET_SOURCE)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an asset from a dictionary. When initializing an asset from a dictionary, All items in "configs" and
        "secrets" will be passed as keyword arguments into the constructor, unless the asset is an instance of the
        generic _Asset class (which is not supposed to be instantiated directly in normal use cases).

        Any secret serialized as a dictionary (i.e. a serialized Haystack ``Secret``) is rehydrated via
        ``Secret.from_dict``. Existing ``Secret`` objects and ``None`` pass through; plain strings are rejected.

        :param data: The dictionary containing the asset
        :return: An instance of the asset
        """
        secrets = normalize_asset_secrets(data)["secrets"]
        if cls.__name__ == "_Asset":
            return cls(
                name=data["name"],
                configs=data.get("configs", {}),
                secrets=secrets,
            )
        return cls(
            name=data.get("name"),
            **data.get("configs") or {},
            **secrets,
        )
