"""
This module provides a collection of classes that represent and manage assets in the application.

An asset is a metadata reference to an external service or resource, which can be utilized across different
applications.
"""

from ravelin.assets._base import _Asset as Asset
from ravelin.assets.resolvers import AssetResolver, normalize_asset_secrets, register_asset_resolver

__all__ = ["Asset", "AssetResolver", "normalize_asset_secrets", "register_asset_resolver"]
