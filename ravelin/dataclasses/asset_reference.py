"""
This module contains the dataclass for AssetReference.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from typing_extensions import Self


@dataclass
class AssetReference:
    """
    A dataclass that represents a hashable reference to an asset. This provides a unified way to identify and
    retrieve an asset from the asset registry, either locally or remotely.

    :param name: The name of the asset.
    :param source: The source of the asset.
    """

    name: str
    source: str = "local"
    version: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the portable reference coordinates without resolving them."""
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Asset reference source must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the asset reference to a dictionary.

        :return: A dictionary representation of the asset reference.
        """
        return {"name": self.name, "source": self.source, "version": self.version}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an AssetReference instance from a dictionary.

        :param data: A dictionary representation of the asset reference.
        :return: An instance of AssetReference.
        """
        return cls(name=data["name"], source=data.get("source", "local"), version=data.get("version"))
